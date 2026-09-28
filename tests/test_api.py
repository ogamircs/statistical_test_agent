"""HTTP/SSE API for the React UI (src/api), exercised with a stub agent."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
import plotly.graph_objects as go
import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.api.progress import format_sse
from src.api.sessions import compose_agent_message, parse_user_message
from src.api.tokens import issue_token, verify_token
from src.config import Config
from src.query_store import SQLiteQueryStore
from src.statistics import ABTestAnalyzer, ABTestVisualizer

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "sample_ab_data.csv"


class _Session:
    def __init__(self, path: str) -> None:
        self.query_store = SQLiteQueryStore(path)


class StubAgent:
    """Mimics the parts of ABTestingAgent the API touches."""

    def __init__(self, query_store_path: str) -> None:
        self.session = _Session(query_store_path)
        self.visualizer = ABTestVisualizer()
        self.charts: Dict[str, go.Figure] = {}
        self._last_results: Optional[List[Any]] = None
        self._last_summary: Any = None
        self.received: List[str] = []
        self.cleared = False
        self.release = threading.Event()
        self.release.set()
        self.response = "## Result\n\n| a | b |\n|---|---|\n| 1 | 2 |"
        self.raise_error = False

    def run(self, message: str, callbacks: Any = None) -> str:
        self.release.wait(5)
        if self.raise_error:
            raise RuntimeError("boom")
        self.received.append(message)
        run_id = uuid4()
        for handler in callbacks or []:
            handler.on_tool_start({"name": "load_and_auto_analyze"}, "{}", run_id=run_id)
            handler.on_tool_end("ok", run_id=run_id)
        self.session.query_store.save_chat_message("human", message)
        self.session.query_store.save_chat_message("ai", self.response)
        self.charts = {"dashboard": go.Figure(layout={"title": {"text": "<b>Dash</b>"}})}
        return self.response

    def get_charts(self) -> Dict[str, go.Figure]:
        return self.charts

    def clear_charts(self) -> None:
        self.charts = {}

    def clear_memory(self) -> None:
        self.cleared = True
        self.session.query_store.clear_chat_messages()

    def _get_active_analyzer(self) -> Any:
        return None


@pytest.fixture
def agents() -> List[StubAgent]:
    return []


@pytest.fixture
def client(tmp_path: Path, agents: List[StubAgent], monkeypatch) -> TestClient:
    monkeypatch.delenv("STATAGENT_AUTH_USERNAME", raising=False)
    monkeypatch.delenv("STATAGENT_AUTH_PASSWORD", raising=False)

    def factory(path: str) -> StubAgent:
        agent = StubAgent(path)
        agents.append(agent)
        return agent

    app = create_app(
        config=Config(max_upload_mb=1),
        agent_factory=factory,
        store_dir=tmp_path / "store",
        uploads_dir=tmp_path / "uploads",
        frontend_dist=tmp_path / "no-dist",
    )
    return TestClient(app)


def _events(body: str) -> List[tuple[str, Dict[str, Any]]]:
    events = []
    for frame in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def _new_session(client: TestClient) -> str:
    response = client.post("/api/sessions")
    assert response.status_code == 201
    return response.json()["id"]


def test_health_and_public_config(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}
    config = client.get("/api/config").json()
    assert config["auth_required"] is False
    assert config["max_upload_mb"] == 1
    keys = {option["key"] for option in config["chart_types"]}
    assert {"dashboard", "all", "bayesian", "p_values", "effect_waterfall"} <= keys


def test_chat_streams_progress_message_charts_and_done(client: TestClient, agents) -> None:
    session_id = _new_session(client)

    response = client.post(f"/api/sessions/{session_id}/chat", json={"message": "analyze"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response.text)
    names = [name for name, _ in events]
    assert names == ["status", "tool_start", "tool_end", "message", "charts", "done"]
    assert events[1][1]["label"] == "Loading data and running the analysis"
    assert events[2][1]["ok"] is True
    assert events[3][1] == {"content": agents[0].response, "error_code": None}
    chart = events[4][1]["charts"][0]
    assert chart["name"] == "dashboard" and chart["title"] == "Dash"
    assert "layout" in chart["figure"]


def test_chat_surfaces_error_code_from_agent_response(client: TestClient, agents) -> None:
    session_id = _new_session(client)
    agents[0].response = "Error processing request: nope [error_code=LLM_TIMEOUT]"

    events = _events(client.post(f"/api/sessions/{session_id}/chat", json={"message": "x"}).text)

    message = dict(events)["message"]
    assert message["error_code"] == "LLM_TIMEOUT"


def test_chat_crash_emits_error_event_and_releases_session(client: TestClient, agents) -> None:
    session_id = _new_session(client)
    agents[0].raise_error = True

    events = _events(client.post(f"/api/sessions/{session_id}/chat", json={"message": "x"}).text)

    assert [name for name, _ in events] == ["status", "error", "done"]
    agents[0].raise_error = False
    assert client.post(f"/api/sessions/{session_id}/chat", json={"message": "again"}).status_code == 200


def test_chat_rejects_empty_message_and_unknown_session(client: TestClient) -> None:
    session_id = _new_session(client)
    empty = client.post(f"/api/sessions/{session_id}/chat", json={"message": "  "})
    assert empty.status_code == 400
    assert empty.json()["error"]["code"] == "EMPTY_MESSAGE"
    missing = client.post("/api/sessions/deadbeefdeadbeef/chat", json={"message": "hi"})
    assert missing.status_code == 404
    assert client.post("/api/sessions/../../etc/chat", json={"message": "hi"}).status_code == 404


def test_upload_returns_preview_and_chat_uses_upload_path(client: TestClient, agents, tmp_path) -> None:
    session_id = _new_session(client)
    with SAMPLE_CSV.open("rb") as handle:
        response = client.post(
            f"/api/sessions/{session_id}/upload",
            files={"file": ("experiment.csv", handle, "text/csv")},
        )

    assert response.status_code == 201
    payload = response.json()
    preview = payload["preview"]
    assert payload["filename"] == "experiment.csv"
    assert preview["row_count"] == len(pd.read_csv(SAMPLE_CSV))
    assert len(preview["rows"]) == 20
    assert {"name", "dtype", "missing_pct"} <= set(preview["columns"][0])

    client.post(
        f"/api/sessions/{session_id}/chat",
        json={"message": "best guess", "file_id": payload["file_id"]},
    )
    sent = agents[0].received[-1]
    stored = tmp_path / "uploads" / session_id / f"{payload['file_id']}.csv"
    assert sent == f"User request: best guess\n\nCSV file path: {stored.resolve()}"

    messages = client.get(f"/api/sessions/{session_id}/messages").json()["messages"]
    assert messages[0] == {"role": "user", "content": "best guess", "attachment": "experiment.csv"}


@pytest.mark.parametrize(
    ("filename", "content", "code"),
    [
        ("data.txt", b"a,b\n1,2\n", "UPLOAD_REJECTED"),
        ("big.csv", b"a,b\n" + b"1,2\n" * 400_000, "UPLOAD_REJECTED"),
        ("empty.csv", b"", "UPLOAD_REJECTED"),
    ],
)
def test_upload_validation(client: TestClient, tmp_path, filename, content, code) -> None:
    session_id = _new_session(client)
    response = client.post(
        f"/api/sessions/{session_id}/upload",
        files={"file": (filename, content, "text/csv")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == code
    leftovers = list((tmp_path / "uploads" / session_id).glob("*.csv"))
    assert leftovers == []


def test_chat_rejects_unknown_or_malformed_file_id(client: TestClient) -> None:
    session_id = _new_session(client)
    for file_id in ("0" * 32, "../../secret"):
        response = client.post(
            f"/api/sessions/{session_id}/chat", json={"message": "x", "file_id": file_id}
        )
        assert response.status_code == 404


def test_session_listing_titles_resume_clear_and_delete(client: TestClient, agents, tmp_path) -> None:
    first = _new_session(client)
    _new_session(client)  # no messages yet -> not listed
    client.post(f"/api/sessions/{first}/chat", json={"message": "How did variant B do?"})

    listed = client.get("/api/sessions").json()["sessions"]
    assert [item["id"] for item in listed] == [first]
    assert listed[0]["title"] == "How did variant B do?"

    history = client.get(f"/api/sessions/{first}/messages").json()
    assert [m["role"] for m in history["messages"]] == ["user", "assistant"]
    assert history["charts"][0]["name"] == "dashboard"

    assert client.delete(f"/api/sessions/{first}/messages").status_code == 204
    assert agents[0].cleared is True
    assert client.get(f"/api/sessions/{first}/messages").json() == {"messages": [], "charts": []}

    assert client.delete(f"/api/sessions/{first}").status_code == 204
    assert not (tmp_path / "store" / f"session-{first}.sqlite").exists()
    assert client.get(f"/api/sessions/{first}/messages").status_code == 404


def test_session_resumes_from_disk_after_restart(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("STATAGENT_AUTH_USERNAME", raising=False)
    kwargs = dict(
        agent_factory=StubAgent,
        store_dir=tmp_path / "store",
        uploads_dir=tmp_path / "uploads",
        frontend_dist=tmp_path / "none",
    )
    before = TestClient(create_app(**kwargs))  # type: ignore[arg-type]
    session_id = before.post("/api/sessions").json()["id"]
    before.post(f"/api/sessions/{session_id}/chat", json={"message": "hello"})

    after = TestClient(create_app(**kwargs))  # type: ignore[arg-type]
    messages = after.get(f"/api/sessions/{session_id}/messages").json()["messages"]
    assert [m["content"] for m in messages][:1] == ["hello"]


def test_busy_session_returns_409(client: TestClient, agents) -> None:
    session_id = _new_session(client)
    record = client.app.state.registry.get(session_id)  # type: ignore[attr-defined]
    assert record.lock.acquire(blocking=False)
    try:
        response = client.post(f"/api/sessions/{session_id}/chat", json={"message": "x"})
    finally:
        record.lock.release()
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "SESSION_BUSY"


def test_charts_endpoint_builds_from_last_analysis(client: TestClient, agents) -> None:
    session_id = _new_session(client)
    assert client.get(f"/api/sessions/{session_id}/charts").status_code == 409

    analyzer = ABTestAnalyzer()
    analyzer.load_data(str(SAMPLE_CSV))
    analyzer.auto_configure()
    results = analyzer.run_segmented_analysis()
    agents[0]._last_results = results
    agents[0]._last_summary = analyzer.generate_summary(results)

    dashboard = client.get(f"/api/sessions/{session_id}/charts", params={"type": "dashboard"})
    assert dashboard.status_code == 200
    assert [c["name"] for c in dashboard.json()["charts"]] == ["dashboard"]

    bayesian = client.get(f"/api/sessions/{session_id}/charts", params={"type": "bayesian"})
    assert len(bayesian.json()["charts"]) == 3

    unknown = client.get(f"/api/sessions/{session_id}/charts", params={"type": "nope"})
    assert unknown.status_code == 400


def test_auth_required_when_credentials_configured(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("STATAGENT_AUTH_USERNAME", "alice")
    monkeypatch.setenv("STATAGENT_AUTH_PASSWORD", "s3cret")

    assert client.get("/api/config").json()["auth_required"] is True
    assert client.get("/api/health").status_code == 200
    assert client.post("/api/sessions").status_code == 401
    assert client.post("/api/login", json={"username": "alice", "password": "bad"}).status_code == 401

    token = client.post("/api/login", json={"username": "alice", "password": "s3cret"}).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert client.post("/api/sessions", headers=headers).status_code == 201
    assert client.get("/api/sessions", headers={"Authorization": "Bearer forged.abc"}).status_code == 401


def test_login_rejected_when_auth_disabled(client: TestClient) -> None:
    response = client.post("/api/login", json={"username": "a", "password": "b"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "AUTH_DISABLED"


def test_tokens_expire_and_reject_tampering() -> None:
    token = issue_token("alice")
    assert verify_token(token) == "alice"
    payload, signature = token.split(".")
    assert verify_token(f"{payload}.{'0' * len(signature)}") is None
    assert verify_token(issue_token("alice", ttl_seconds=-1)) is None
    assert verify_token("garbage") is None


def test_frontend_served_with_spa_fallback(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>app</html>", encoding="utf-8")
    (dist / "assets" / "main.js").write_text("console.log(1)", encoding="utf-8")
    client = TestClient(
        create_app(agent_factory=StubAgent, store_dir=tmp_path / "s", frontend_dist=dist)
    )

    assert client.get("/").text == "<html>app</html>"
    assert client.get("/sessions/abc").text == "<html>app</html>"
    assert client.get("/assets/main.js").text == "console.log(1)"
    assert client.get("/../../etc/passwd").text == "<html>app</html>"
    assert client.get("/api/unknown").status_code == 404


def test_message_helpers_round_trip() -> None:
    composed = compose_agent_message("  hi  ", "/x/abc.csv")
    assert parse_user_message(composed, {"abc": "real.csv"}) == {
        "role": "user",
        "content": "hi",
        "attachment": "real.csv",
    }
    assert parse_user_message(compose_agent_message("", "/x/abc.csv"))["content"] == ""
    assert compose_agent_message("plain", None) == "plain"
    assert format_sse("done", {}) == "event: done\ndata: {}\n\n"


def test_app_module_exposes_fastapi_app() -> None:
    from app import app

    assert TestClient(app).get("/api/health").status_code == 200


# -- review fixes: session lock covers every state-touching endpoint --------


@pytest.mark.parametrize(
    "method, path",
    [
        ("DELETE", "/api/sessions/{id}"),
        ("DELETE", "/api/sessions/{id}/messages"),
        ("GET", "/api/sessions/{id}/charts"),
    ],
)
def test_state_endpoints_reject_while_a_run_is_active(
    client: TestClient, agents, tmp_path, method: str, path: str
) -> None:
    session_id = _new_session(client)
    client.post(f"/api/sessions/{session_id}/chat", json={"message": "hello"})
    registry = client.app.state.registry  # type: ignore[attr-defined]
    record = registry.get(session_id)
    assert record.lock.acquire(blocking=False)  # simulate an in-flight run
    try:
        response = client.request(method, path.format(id=session_id))
    finally:
        record.lock.release()

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "SESSION_BUSY"
    # Nothing was cleared or deleted underneath the run.
    assert agents[0].cleared is False
    assert registry.store_path(session_id).exists()
    assert client.get(f"/api/sessions/{session_id}/messages").json()["messages"]


def test_on_demand_charts_become_the_sessions_latest_charts(client: TestClient, agents) -> None:
    session_id = _new_session(client)
    analyzer = ABTestAnalyzer()
    analyzer.load_data(str(SAMPLE_CSV))
    analyzer.auto_configure()
    results = analyzer.run_segmented_analysis()
    agents[0]._last_results = results
    agents[0]._last_summary = analyzer.generate_summary(results)

    client.get(f"/api/sessions/{session_id}/charts", params={"type": "bayesian"})

    reopened = client.get(f"/api/sessions/{session_id}/messages").json()["charts"]
    assert len(reopened) == 3
    assert all(chart["name"].startswith("bayesian") for chart in reopened)


def test_charts_are_finalized_even_if_the_client_disconnects(client: TestClient, agents) -> None:
    session_id = _new_session(client)
    agent = agents[0]
    agent.release.clear()  # hold the run until the client has gone away

    with client.stream("POST", f"/api/sessions/{session_id}/chat", json={"message": "go"}) as stream:
        first = next(stream.iter_lines())
        assert first.startswith("event: status")
    agent.release.set()

    record = client.app.state.registry.get(session_id)  # type: ignore[attr-defined]
    for _ in range(100):
        if record.lock.acquire(blocking=False):
            record.lock.release()
            if record.latest_charts:
                break
        threading.Event().wait(0.05)

    assert [chart["name"] for chart in record.latest_charts] == ["dashboard"]
    # Cleared from the agent, so a later text-only turn cannot re-emit them.
    assert agent.charts == {}


# -- STATAGENT_REQUIRE_AUTH (TODO.md #49) -----------------------------------


def test_require_auth_refuses_to_start_without_credentials(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("STATAGENT_AUTH_USERNAME", raising=False)
    monkeypatch.delenv("STATAGENT_AUTH_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="STATAGENT_REQUIRE_AUTH"):
        create_app(config=Config(require_auth=True), store_dir=tmp_path, uploads_dir=tmp_path)


def test_require_auth_starts_and_enforces_when_credentials_set(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("STATAGENT_AUTH_USERNAME", "ana")
    monkeypatch.setenv("STATAGENT_AUTH_PASSWORD", "s3cret")
    app = create_app(
        config=Config(require_auth=True),
        agent_factory=StubAgent,
        store_dir=tmp_path / "store",
        uploads_dir=tmp_path / "uploads",
        frontend_dist=tmp_path / "no-dist",
    )
    client = TestClient(app)
    assert client.get("/api/sessions").status_code == 401
    assert client.get("/api/config").json()["auth_required"] is True
