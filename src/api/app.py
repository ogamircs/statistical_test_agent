"""FastAPI application: REST + SSE endpoints for the React UI.

SSE contract for ``POST /api/sessions/{id}/chat`` (``text/event-stream``),
in order:

- ``status``      ``{"state": "started"}``
- ``tool_start``  ``{"id", "name", "label"}`` (zero or more)
- ``tool_end``    ``{"id", "name", "ok"}``    (zero or more)
- ``token``       ``{"text": str}`` (zero or more, interleaved with tool events;
  model text as generated, reset by the client on each ``tool_start``)
- ``message``     ``{"content": markdown, "error_code": str | null}`` (the full
  final answer; replaces any streamed text)
- ``charts``      ``{"charts": [{"name", "title", "figure"}], "state": str}``
  (figure = Plotly JSON). ``state`` tells the client what to do with the
  charts it shows: ``"updated"`` replace them with ``charts``; ``"cleared"``
  the analysis they described was replaced (new data or re-run without
  charting), remove them (``charts`` is empty); ``"unchanged"`` keep them.
- ``done``        ``{}``

On an unexpected failure an ``error`` event ``{"code", "message"}`` replaces
``message``/``charts`` and is followed by ``done``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Tuple

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.auth import is_auth_enabled, verify_credentials
from src.config import Config
from src.data_paths import UPLOADS_DIRNAME

from .charts import (
    CHART_TYPE_OPTIONS,
    NoAnalysisError,
    UnknownChartTypeError,
    build_charts_for_agent,
    serialize_charts,
)
from .progress import ToolProgressHandler, format_sse
from .sessions import (
    AgentFactory,
    SessionNotFoundError,
    SessionRegistry,
    UploadNotFoundError,
    UploadRejectedError,
    analysis_version,
    compose_agent_message,
    remember_charts,
)
from .tokens import TokenSigner

logger = logging.getLogger(__name__)

_ERROR_CODE = re.compile(r"\[error_code=([A-Z0-9_]+)\]")
_UPLOAD_CHUNK_BYTES = 1024 * 1024


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=20_000)
    file_id: Optional[str] = None


def _default_agent_factory(config: Config) -> AgentFactory:
    def factory(query_store_path: str) -> Any:
        from src.agent import ABTestingAgent

        return ABTestingAgent(config=config, query_store_path=query_store_path)

    return factory


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def create_app(
    *,
    config: Optional[Config] = None,
    agent_factory: Optional[AgentFactory] = None,
    store_dir: Optional[Path] = None,
    uploads_dir: Optional[Path] = None,
    frontend_dist: Optional[Path] = None,
) -> FastAPI:
    config = config or Config.from_env()
    try:
        # Fail fast: a deployment that asked for auth must never run open.
        config.validate_security()
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    # Everything auth-related comes from the resolved Config, so an injected
    # config (tests, embedding) fully controls credentials and signing key.
    auth_enabled = is_auth_enabled(config)
    signer = TokenSigner(config.auth_secret)
    resolved_uploads = uploads_dir or Path.cwd() / UPLOADS_DIRNAME
    # Agents must be allowed to load what this app stores as uploads, even
    # when uploads_dir is not the default <cwd>/.uploads.
    agent_config = dataclasses.replace(
        config, data_roots=(*config.data_roots, str(resolved_uploads))
    )
    registry = SessionRegistry(
        store_dir=store_dir or Path(config.query_store_dir),
        uploads_dir=resolved_uploads,
        agent_factory=agent_factory or _default_agent_factory(agent_config),
        max_upload_bytes=int(config.max_upload_mb * 1024 * 1024),
    )
    app = FastAPI(title="A/B Testing Agent", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.registry = registry

    def require_auth(request: Request) -> Optional[str]:
        if not auth_enabled:
            return None
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        user = signer.verify(token) if scheme.lower() == "bearer" and token else None
        if user is None:
            raise _error(401, "AUTH_REQUIRED", "Sign in to use the agent.")
        return user

    @contextmanager
    def idle_session(record: Any) -> Iterator[None]:
        """Hold the session lock, or fail fast with SESSION_BUSY.

        Every endpoint that mutates or reads the agent's analysis state takes
        the same lock as an active chat run, so clearing, deleting or charting
        can never interleave with a run (orphaned turns, deleted uploads mid
        run, charts mixing a new dataframe with old results).
        """
        if not record.lock.acquire(blocking=False):
            raise _error(409, "SESSION_BUSY", "This session is still working on a request.")
        try:
            yield
        finally:
            record.lock.release()

    def session_or_404(session_id: str) -> Any:
        try:
            return registry.get(session_id)
        except SessionNotFoundError as error:
            raise _error(404, "SESSION_NOT_FOUND", "Unknown session.") from error

    # -- public endpoints --------------------------------------------------

    @app.get("/api/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/config")
    def public_config() -> Dict[str, Any]:
        return {
            "auth_required": auth_enabled,
            "max_upload_mb": config.max_upload_mb,
            "chart_types": CHART_TYPE_OPTIONS,
            "model": config.llm_model,
        }

    @app.post("/api/login")
    def login(body: LoginRequest) -> Dict[str, str]:
        if not auth_enabled:
            raise _error(400, "AUTH_DISABLED", "Authentication is not enabled on this server.")
        user = verify_credentials(config, body.username, body.password)
        if user is None:
            raise _error(401, "INVALID_CREDENTIALS", "Invalid username or password.")
        return {"token": signer.issue(user)}

    # -- sessions ----------------------------------------------------------

    auth = [Depends(require_auth)]

    @app.get("/api/sessions", dependencies=auth)
    def list_sessions() -> Dict[str, Any]:
        return {"sessions": registry.list_sessions()}

    @app.post("/api/sessions", dependencies=auth, status_code=201)
    def create_session() -> Dict[str, str]:
        return {"id": registry.create()}

    @app.delete("/api/sessions/{session_id}", dependencies=auth, status_code=204)
    def delete_session(session_id: str) -> None:
        record = session_or_404(session_id)
        with idle_session(record):
            try:
                registry.delete(session_id)
            except SessionNotFoundError as error:
                raise _error(404, "SESSION_NOT_FOUND", "Unknown session.") from error

    @app.get("/api/sessions/{session_id}/messages", dependencies=auth)
    def get_messages(session_id: str) -> Dict[str, Any]:
        record = session_or_404(session_id)
        return {"messages": registry.messages(session_id), "charts": record.latest_charts}

    @app.delete("/api/sessions/{session_id}/messages", dependencies=auth, status_code=204)
    def clear_messages(session_id: str) -> None:
        record = session_or_404(session_id)
        with idle_session(record):
            record.agent.clear_memory()
            record.agent.clear_charts()
            remember_charts(record, [])

    @app.post("/api/sessions/{session_id}/upload", dependencies=auth, status_code=201)
    def upload(session_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
        record = session_or_404(session_id)

        def chunks() -> Any:
            while True:
                chunk = file.file.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    return
                yield chunk

        # Same lock as runs/delete: an upload can't race a session delete
        # (orphaned directories, 500s) or another upload's names.json update.
        with idle_session(record):
            try:
                return registry.save_upload(session_id, file.filename or "", chunks())
            except UploadRejectedError as error:
                raise _error(400, "UPLOAD_REJECTED", str(error)) from error

    @app.get("/api/sessions/{session_id}/charts", dependencies=auth)
    def charts(session_id: str, type: str = Query("dashboard", max_length=64)) -> Dict[str, Any]:
        record = session_or_404(session_id)
        with idle_session(record):
            try:
                built = build_charts_for_agent(record.agent, type)
            except NoAnalysisError as error:
                raise _error(409, "NO_ANALYSIS", str(error)) from error
            except UnknownChartTypeError as error:
                raise _error(400, "UNKNOWN_CHART_TYPE", str(error)) from error
            # Persist the selection so reopening the session shows these charts.
            remember_charts(record, built)
        return {"charts": built}

    @app.post("/api/sessions/{session_id}/chat", dependencies=auth)
    async def chat(session_id: str, body: ChatRequest) -> StreamingResponse:
        record = session_or_404(session_id)
        file_path: Optional[str] = None
        if body.file_id:
            try:
                file_path = str(registry.resolve_upload(session_id, body.file_id))
            except UploadNotFoundError as error:
                raise _error(404, "UPLOAD_NOT_FOUND", "Unknown uploaded file.") from error
        agent_message = compose_agent_message(body.message, file_path)
        if not agent_message:
            raise _error(400, "EMPTY_MESSAGE", "Type a message or attach a CSV file.")
        if not record.lock.acquire(blocking=False):
            raise _error(409, "SESSION_BUSY", "This session is still working on a request.")

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Dict[str, Any]]] = asyncio.Queue()

        def emit(event: str, data: Dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, (event, data))

        def run() -> Tuple[str, Dict[str, Any]]:
            try:
                response = str(
                    record.agent.run(
                        agent_message,
                        callbacks=[ToolProgressHandler(emit)],
                        on_token=lambda text: emit("token", {"text": text}),
                    )
                )
                # Finalize charts here, still under the lock, so a client that
                # disconnects mid-stream cannot strand them in the agent (where
                # a later turn would re-emit them as its own).
                return response, _finalize_charts(record)
            finally:
                record.lock.release()

        # Start the run here, not inside the stream: the lock is then always
        # released even if the client disconnects before streaming begins,
        # and a finished run still lands in the persisted history.
        task = loop.run_in_executor(None, run)
        return StreamingResponse(
            _chat_events(record, queue, task),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def _chat_events(
        record: Any,
        queue: "asyncio.Queue[tuple[str, Dict[str, Any]]]",
        task: "asyncio.Future[Tuple[str, Dict[str, Any]]]",
    ) -> AsyncIterator[str]:
        yield format_sse("status", {"state": "started"})
        while True:
            getter = asyncio.ensure_future(queue.get())
            pending: set[asyncio.Future[Any]] = {getter, task}
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            if getter in done:
                event, data = getter.result()
                yield format_sse(event, data)
                continue
            getter.cancel()
            break
        while not queue.empty():
            event, data = queue.get_nowait()
            yield format_sse(event, data)

        try:
            response, charts_event = task.result()
        except Exception:
            logger.exception("Agent run crashed")
            yield format_sse(
                "error",
                {"code": "UI_MESSAGE_HANDLING_FAILED", "message": "Unable to handle this request right now."},
            )
            yield format_sse("done", {})
            return

        match = _ERROR_CODE.search(response)
        yield format_sse("message", {"content": response, "error_code": match.group(1) if match else None})
        yield format_sse("charts", charts_event)
        yield format_sse("done", {})

    # -- frontend ----------------------------------------------------------

    dist = frontend_dist if frontend_dist is not None else Path("frontend") / "dist"
    if dist.is_dir():
        _mount_frontend(app, dist.resolve())

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, dict) else {"code": "HTTP_ERROR", "message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content={"error": detail})

    return app


def _finalize_charts(record: Any) -> Dict[str, Any]:
    """Settle the session's charts after a run (worker thread, under the lock).

    Returns the ``charts`` SSE payload. New charts replace the stored ones;
    if the run replaced the analysis without charting it, the stored charts
    are dropped so neither the live UI nor a reopened session shows results
    for a superseded dataset/analysis.
    """
    charts = record.agent.get_charts()
    if charts:
        record.agent.clear_charts()
        try:
            serialized = serialize_charts(charts)
        except Exception:
            logger.exception("Chart serialization failed; sending the reply without charts")
            serialized = []
        if serialized:
            remember_charts(record, serialized)
            return {"charts": serialized, "state": "updated"}
    if record.charts_version != analysis_version(record.agent):
        remember_charts(record, [])
        return {"charts": [], "state": "cleared"}
    return {"charts": [], "state": "unchanged"}


def _mount_frontend(app: FastAPI, dist: Path) -> None:
    """Serve the built SPA: real files when they exist, else index.html."""
    index = dist / "index.html"

    def serve(path: str) -> FileResponse:
        candidate = (dist / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(dist):
            return FileResponse(candidate)
        return FileResponse(index)

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise _error(404, "NOT_FOUND", "Unknown API route.")
        return serve(path)
