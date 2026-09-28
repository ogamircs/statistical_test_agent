"""Server-side chat sessions: one ``ABTestingAgent`` per session.

Each session persists its chat history in
``<store_dir>/session-<id>.sqlite`` (the agent's query store), so the
history sidebar can list and resume sessions across server restarts.
Uploaded CSVs live in ``<uploads_dir>/<id>/<file_id>.csv``.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

import pandas as pd

from src.query_store import SQLiteQueryStore

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9-]{8,64}$")
FILE_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
PREVIEW_ROWS = 20

_UPLOAD_MESSAGE = re.compile(r"^User request: (?P<request>.*)\n\nCSV file path: (?P<path>.+)$", re.S)
_LOAD_ONLY_MESSAGE = re.compile(r"^Load the CSV file at path: (?P<path>.+)$", re.S)

AgentFactory = Callable[[str], Any]


class SessionNotFoundError(LookupError):
    pass


class UploadNotFoundError(LookupError):
    pass


class UploadRejectedError(ValueError):
    pass


def compose_agent_message(message: str, file_path: Optional[str]) -> str:
    """Wording the agent sees for a message with an attached CSV."""
    text = message.strip()
    if file_path is None:
        return text
    if text:
        return f"User request: {text}\n\nCSV file path: {file_path}"
    return f"Load the CSV file at path: {file_path}"


def parse_user_message(content: str, names: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Split a persisted human message back into text + attachment name.

    The envelope is only decoded when its path names a file this session
    actually uploaded (``names`` maps upload file ids to original names).
    Otherwise the message is user-typed text that merely looks like the
    envelope, and it is shown verbatim rather than inventing an attachment.
    """
    uploads = names or {}
    for pattern in (_UPLOAD_MESSAGE, _LOAD_ONLY_MESSAGE):
        match = pattern.match(content)
        if not match:
            continue
        stem = Path(match.group("path").strip()).stem
        if not FILE_ID_PATTERN.match(stem) or stem not in uploads:
            break
        text = match.groupdict().get("request") or ""
        return {"role": "user", "content": text.strip(), "attachment": uploads[stem]}
    return {"role": "user", "content": content, "attachment": None}


@dataclass
class SessionRecord:
    agent: Any
    lock: threading.Lock = field(default_factory=threading.Lock)
    latest_charts: List[Dict[str, Any]] = field(default_factory=list)


_LATEST_CHARTS_KEY = "latest_charts"
_charts_logger = logging.getLogger(__name__)


def remember_charts(record: SessionRecord, charts: List[Dict[str, Any]]) -> None:
    """Set the session's latest charts and persist them for restart recovery."""
    record.latest_charts = charts
    try:
        record.agent.session.query_store.save_state(_LATEST_CHARTS_KEY, charts)
    except Exception:
        _charts_logger.exception("Failed to persist latest charts; they will not survive a restart")


def _load_persisted_charts(agent: Any) -> List[Dict[str, Any]]:
    try:
        charts = agent.session.query_store.load_state(_LATEST_CHARTS_KEY)
    except Exception:
        _charts_logger.exception("Failed to read persisted charts")
        return []
    return charts if isinstance(charts, list) else []


class SessionRegistry:
    """Owns the live agents and the on-disk session/upload layout."""

    def __init__(
        self,
        *,
        store_dir: Path,
        uploads_dir: Path,
        agent_factory: AgentFactory,
        max_upload_bytes: int,
    ) -> None:
        self.store_dir = Path(store_dir)
        self.uploads_dir = Path(uploads_dir)
        self.agent_factory = agent_factory
        self.max_upload_bytes = max_upload_bytes
        self._records: Dict[str, SessionRecord] = {}
        self._guard = threading.Lock()

    # -- session lifecycle -------------------------------------------------

    def store_path(self, session_id: str) -> Path:
        return self.store_dir / f"session-{session_id}.sqlite"

    def _validate_id(self, session_id: str) -> None:
        if not SESSION_ID_PATTERN.match(session_id):
            raise SessionNotFoundError(session_id)

    def create(self) -> str:
        session_id = uuid4().hex
        self.get(session_id, create=True)
        return session_id

    def get(self, session_id: str, *, create: bool = False) -> SessionRecord:
        self._validate_id(session_id)
        with self._guard:
            record = self._records.get(session_id)
            if record is not None:
                return record
            if not create and not self.store_path(session_id).exists():
                raise SessionNotFoundError(session_id)
            self.store_dir.mkdir(parents=True, exist_ok=True)
            agent = self.agent_factory(str(self.store_path(session_id)))
            record = SessionRecord(agent=agent, latest_charts=_load_persisted_charts(agent))
            self._records[session_id] = record
            return record

    def delete(self, session_id: str) -> None:
        self._validate_id(session_id)
        with self._guard:
            self._records.pop(session_id, None)
        for suffix in ("", "-wal", "-shm"):
            path = Path(f"{self.store_path(session_id)}{suffix}")
            if path.exists():
                path.unlink()
        shutil.rmtree(self.uploads_dir / session_id, ignore_errors=True)

    def list_sessions(self) -> List[Dict[str, Any]]:
        sessions: List[Dict[str, Any]] = []
        if not self.store_dir.exists():
            return sessions
        for path in self.store_dir.glob("session-*.sqlite"):
            session_id = path.stem[len("session-"):]
            if not SESSION_ID_PATTERN.match(session_id):
                continue
            messages = SQLiteQueryStore(path).load_chat_messages()
            first_human = next((m for m in messages if m.get("role") == "human"), None)
            if first_human is None:
                continue
            parsed = parse_user_message(first_human.get("content", ""), self._upload_names(session_id))
            title = parsed["content"] or (
                f"Analyze {parsed['attachment']}" if parsed["attachment"] else "New analysis"
            )
            sessions.append(
                {
                    "id": session_id,
                    "title": title[:120],
                    "updated_at": datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "message_count": len(messages),
                }
            )
        sessions.sort(key=lambda item: item["updated_at"], reverse=True)
        return sessions

    def messages(self, session_id: str) -> List[Dict[str, Any]]:
        record = self.get(session_id)
        names = self._upload_names(session_id)
        output: List[Dict[str, Any]] = []
        for entry in record.agent.session.query_store.load_chat_messages():
            content = entry.get("content", "")
            if entry.get("role") == "human":
                output.append(parse_user_message(content, names))
            elif entry.get("role") == "ai":
                output.append({"role": "assistant", "content": content, "attachment": None})
        return output

    # -- uploads -----------------------------------------------------------

    def _upload_names(self, session_id: str) -> Dict[str, str]:
        index = self.uploads_dir / session_id / "names.json"
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def save_upload(self, session_id: str, filename: str, chunks: Any) -> Dict[str, Any]:
        """Stream an upload to disk, enforcing extension and size limits."""
        self.get(session_id)
        original = Path(filename or "").name
        if not original.lower().endswith(".csv"):
            raise UploadRejectedError("Only .csv files can be uploaded.")

        session_dir = self.uploads_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        file_id = uuid4().hex
        target = session_dir / f"{file_id}.csv"
        written = 0
        try:
            with target.open("wb") as handle:
                for chunk in chunks:
                    written += len(chunk)
                    if written > self.max_upload_bytes:
                        raise UploadRejectedError(
                            f"File exceeds the {self.max_upload_bytes / (1024 * 1024):.0f} MB upload limit."
                        )
                    handle.write(chunk)
            preview = build_preview(target)
        except Exception:
            target.unlink(missing_ok=True)
            raise

        names = self._upload_names(session_id)
        names[file_id] = original
        (session_dir / "names.json").write_text(json.dumps(names), encoding="utf-8")
        return {"file_id": file_id, "filename": original, "size_bytes": written, "preview": preview}

    def resolve_upload(self, session_id: str, file_id: str) -> Path:
        if not FILE_ID_PATTERN.match(file_id or ""):
            raise UploadNotFoundError(file_id)
        path = self.uploads_dir / session_id / f"{file_id}.csv"
        if not path.is_file():
            raise UploadNotFoundError(file_id)
        return path.resolve()


def build_preview(path: Path) -> Dict[str, Any]:
    """Row count, dtypes, missing % and the first rows of an uploaded CSV."""
    try:
        df = pd.read_csv(path)
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as error:
        raise UploadRejectedError(f"Could not parse the CSV file: {error}") from error
    if df.columns.empty:
        raise UploadRejectedError("The CSV file has no columns.")
    row_count = int(len(df))
    missing = df.isna().mean() * 100 if row_count else pd.Series(0.0, index=df.columns)
    return {
        "row_count": row_count,
        "columns": [
            {
                "name": str(column),
                "dtype": str(df[column].dtype),
                "missing_pct": round(float(missing[column]), 2),
            }
            for column in df.columns
        ],
        "rows": json.loads(df.head(PREVIEW_ROWS).to_json(orient="records", date_format="iso")),
    }
