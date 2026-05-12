#!/usr/bin/env python3
"""
Ambient context tool.

Reads persisted activity descriptions from the ambient ingest session.
Ingest-time hooks analyze screenshots with vision AI, store only the resulting
description, and discard the source image before reads happen.
"""

import base64
import hashlib
import json
import logging
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Dict, List, Optional

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

AMBIENT_DEFAULT_SESSION_ID = "ambient:journal:context"
_CONTENT_JSON_PREFIX = "\x00json:"


class AmbientIngestValidationError(ValueError):
    """Raised when an ingest payload cannot be analyzed safely."""


class AmbientAnalysisIncompleteError(RuntimeError):
    """Raised when the vision model returns an obviously incomplete final answer."""


def _cache_db_path() -> Path:
    return get_hermes_home() / "cache" / "ambient_context.db"


def _parse_timestamp_bound(value: Any, name: str) -> Optional[float]:
    """Parse a Unix timestamp or timezone-aware ISO-8601/RFC3339 timestamp."""
    if value is None or value == "":
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        raise ValueError(f"{name} must be a Unix timestamp or timezone-aware ISO-8601 string")

    stripped = value.strip()
    try:
        return float(stripped)
    except ValueError:
        pass

    normalized = stripped
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a Unix timestamp or timezone-aware ISO-8601 string"
        ) from exc

    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(
            f"{name} must include a timezone offset, such as 'Z' or '+05:30', "
            "to match ingest timestamps safely"
        )

    return dt.timestamp()


def _init_cache_db() -> None:
    db = _cache_db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type = 'table' AND name = 'image_analyses'
        """
    )
    table_exists = cursor.fetchone() is not None
    if table_exists:
        cursor.execute("PRAGMA table_info(image_analyses)")
        columns = {row[1] for row in cursor.fetchall()}
        if "image_index" not in columns:
            cursor.execute("ALTER TABLE image_analyses RENAME TO image_analyses_legacy")
            table_exists = False
        else:
            if "source_role" not in columns:
                cursor.execute("ALTER TABLE image_analyses ADD COLUMN source_role TEXT")
            if "content_text" not in columns:
                cursor.execute("ALTER TABLE image_analyses ADD COLUMN content_text TEXT")

    if not table_exists:
        cursor.execute(
            """
            CREATE TABLE image_analyses (
                message_id INTEGER NOT NULL,
                image_index INTEGER NOT NULL DEFAULT 0,
                image_hash TEXT NOT NULL,
                session_id TEXT,
                timestamp REAL,
                source_role TEXT,
                content_text TEXT,
                analysis TEXT NOT NULL,
                analyzed_at REAL DEFAULT (unixepoch()),
                PRIMARY KEY (message_id, image_index)
            )
            """
        )

        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'image_analyses_legacy'
            """
        )
        if cursor.fetchone() is not None:
            cursor.execute("PRAGMA table_info(image_analyses_legacy)")
            legacy_columns = {row[1] for row in cursor.fetchall()}
            if {"image_hash", "analysis"}.issubset(legacy_columns):
                message_expr = "message_id" if "message_id" in legacy_columns else "NULL"
                session_expr = "session_id" if "session_id" in legacy_columns else "NULL"
                timestamp_expr = "timestamp" if "timestamp" in legacy_columns else "unixepoch()"
                analyzed_at_expr = "analyzed_at" if "analyzed_at" in legacy_columns else "unixepoch()"
                cursor.execute(
                    f"""
                    SELECT rowid, {message_expr}, image_hash, {session_expr},
                           {timestamp_expr}, analysis, {analyzed_at_expr}
                    FROM image_analyses_legacy
                    """
                )
                legacy_rows = cursor.fetchall()
                used_keys = set()
                cursor.execute("SELECT COALESCE(MAX(message_id), 0) FROM image_analyses")
                next_id = int(cursor.fetchone()[0] or 0) + 1
                for row in legacy_rows:
                    rowid, legacy_message_id, image_hash, session_id, timestamp, analysis, analyzed_at = row
                    event_id = legacy_message_id or rowid or next_id
                    try:
                        event_id = int(event_id)
                    except (TypeError, ValueError):
                        event_id = next_id
                    while (event_id, 0) in used_keys:
                        event_id = next_id
                        next_id += 1
                    used_keys.add((event_id, 0))
                    next_id = max(next_id, event_id + 1)
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO image_analyses
                        (message_id, image_index, image_hash, session_id, timestamp, analysis, analyzed_at)
                        VALUES (?, 0, ?, ?, ?, ?, COALESCE(?, unixepoch()))
                        """,
                        (event_id, image_hash, session_id, timestamp, analysis, analyzed_at),
                    )
            cursor.execute("DROP TABLE image_analyses_legacy")

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_img_hash
        ON image_analyses(image_hash)
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_img_ts
        ON image_analyses(timestamp DESC)
        """
    )
    conn.commit()
    conn.close()


def store_ambient_ingest_events(
    *,
    session_id: str,
    analyses: List[Dict[str, Any]],
) -> List[int]:
    """Persist analyzed ambient events without retaining source images."""
    if not analyses:
        return []

    _init_cache_db()
    conn = sqlite3.connect(str(_cache_db_path()))
    cursor = conn.cursor()
    inserted_ids: List[int] = []
    try:
        cursor.execute("BEGIN IMMEDIATE")
        cursor.execute("SELECT COALESCE(MAX(message_id), 0) FROM image_analyses")
        next_id = int(cursor.fetchone()[0] or 0) + 1
        for item in analyses:
            event_id = next_id
            next_id += 1
            cursor.execute(
                """
                INSERT INTO image_analyses
                (message_id, image_index, image_hash, session_id, timestamp,
                 source_role, content_text, analysis)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    item.get("image_index", 0),
                    item["image_hash"],
                    item.get("session_id") or session_id,
                    item["timestamp"],
                    item.get("role"),
                    item.get("content_text"),
                    item["analysis"],
                ),
            )
            inserted_ids.append(event_id)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return inserted_ids


def _get_ambient_analysis_events(
    limit: int = 20,
    session_id: str = AMBIENT_DEFAULT_SESSION_ID,
    start_timestamp: Optional[float] = None,
    end_timestamp: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Return persisted ambient analysis rows from ambient_context.db."""
    _init_cache_db()
    where_clauses = ["(session_id = ? OR (? = ? AND (session_id IS NULL OR session_id = '')))"]
    params: List[Any] = [session_id, session_id, AMBIENT_DEFAULT_SESSION_ID]
    if start_timestamp is not None:
        where_clauses.append("timestamp >= ?")
        params.append(start_timestamp)
    if end_timestamp is not None:
        where_clauses.append("timestamp <= ?")
        params.append(end_timestamp)
    params.append(limit)

    conn = sqlite3.connect(str(_cache_db_path()))
    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT message_id, image_index, image_hash, session_id, timestamp,
               source_role, content_text, analysis, analyzed_at
        FROM image_analyses
        WHERE {' AND '.join(where_clauses)}
        ORDER BY timestamp DESC, analyzed_at DESC
        LIMIT ?
        """,
        params,
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "message_id": row[0],
            "event_id": row[0],
            "image_index": row[1],
            "image_hash": row[2],
            "session_id": row[3] or session_id,
            "timestamp": row[4],
            "role": row[5],
            "content_text": row[6] or "",
            "analysis": row[7],
            "analyzed_at": row[8],
        }
        for row in rows
    ]


def _extract_images_from_content(
    *,
    message_id: int,
    role: str,
    content: Any,
    timestamp: float,
    reject_unsupported_images: bool = False,
) -> List[Dict[str, Any]]:
    if not content:
        return []

    if isinstance(content, str):
        try:
            raw = content
            if raw.startswith(_CONTENT_JSON_PREFIX):
                raw = raw[len(_CONTENT_JSON_PREFIX):]

            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
    else:
        data = content

    blocks = data if isinstance(data, list) else [data]
    if isinstance(data, dict) and "content" in data:
        blocks = data["content"]
        if isinstance(blocks, str):
            blocks = [{"type": "text", "text": blocks}]
    elif isinstance(data, dict):
        blocks = [data]

    images = []
    text_parts = []
    image_index = 0
    for block in blocks:
        if not isinstance(block, dict):
            continue

        block_type = str(block.get("type", "")).strip().lower()
        if block_type in {"text", "input_text", "output_text"}:
            text_parts.append(block.get("text", ""))
            continue

        if block_type not in {"image_url", "input_image"}:
            continue

        image_ref = block.get("image_url")
        if isinstance(image_ref, dict):
            url = image_ref.get("url", "")
        else:
            url = image_ref or block.get("url", "")
        if not isinstance(url, str) or not url.strip():
            if reject_unsupported_images:
                raise AmbientIngestValidationError("Image block is missing a non-empty image URL")
            continue
        url = url.strip()

        if not url.startswith("data:image/"):
            if reject_unsupported_images:
                raise AmbientIngestValidationError(
                    "Ambient ingest only supports inline data:image/...;base64 image URLs"
                )
            continue

        try:
            header, image_data = url.split(",", 1)
            header_parts = header.split(";")
            mime = header_parts[0].split(":", 1)[1]
            if not any(part.lower() == "base64" for part in header_parts[1:]):
                raise AmbientIngestValidationError("Image data URL must be base64-encoded")
            decoded = base64.b64decode(image_data, validate=True)
            if not decoded:
                raise AmbientIngestValidationError("Image data URL is empty")
            image_hash = hashlib.sha256(decoded).hexdigest()
            images.append(
                {
                    "message_id": message_id,
                    "image_index": image_index,
                    "timestamp": timestamp,
                    "role": role,
                    "content_text": "\n".join(text_parts).strip(),
                    "image_data": image_data,
                    "image_mime": mime,
                    "image_hash": image_hash,
                }
            )
            image_index += 1
        except AmbientIngestValidationError:
            raise
        except Exception as exc:
            if reject_unsupported_images:
                raise AmbientIngestValidationError(
                    f"Failed to parse image data URL: {exc}"
                ) from exc
            logger.warning("Failed to parse data URL in message %s", message_id)

    return images


def _ambient_analysis_looks_incomplete(analysis: str) -> bool:
    """Detect final-answer fragments that should not be persisted."""
    text = analysis.strip()
    if not text:
        return True
    if len(text) < 40:
        return True
    return text[-1] not in ".!?)]}'\""


async def _call_vision_for_activity_log(image_path: Path, prompt: str) -> str:
    from tools.vision_tools import vision_analyze_tool

    result_json = await vision_analyze_tool(
        image_url=str(image_path),
        user_prompt=prompt,
    )
    try:
        result = json.loads(result_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Vision analysis returned invalid JSON") from exc

    if not isinstance(result, dict):
        raise RuntimeError("Vision analysis returned an invalid response shape")
    if not result.get("success", False):
        raise RuntimeError(
            result.get("error")
            or result.get("analysis")
            or "vision analysis failed"
        )

    analysis = str(result.get("analysis") or "").strip()
    if not analysis:
        raise RuntimeError("Vision analysis returned no final content")
    if _ambient_analysis_looks_incomplete(analysis):
        raise AmbientAnalysisIncompleteError(
            "Vision analysis returned an incomplete final content fragment"
        )
    return analysis


async def _analyze_screenshot(
    image_path: Path,
    timestamp: float,
    content_text: str,
) -> str:
    """Call the vision model with a prompt tuned for screenshot activity logging."""
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone()
    time_str = dt.strftime("%Y-%m-%d %H:%M:%S %Z")

    prompt = (
        "This is a screenshot captured from Aditya's laptop. "
        f"It was taken at {time_str}.\n\n"
        "Your job is to produce a definitive, concise activity log entry describing "
        "EXACTLY what Aditya was doing at that moment.\n\n"
        "Instructions:\n"
        "1. Identify the primary active application, window, browser tab, or game.\n"
        "2. Note any visible code editors, IDEs, terminals, or file names.\n"
        "3. If a browser is open, identify the website, page title, and any visible content.\n"
        "4. Note any videos, streams, or media playing.\n"
        "5. If chat or messaging apps are visible, note which ones.\n"
        "6. Mention if this appears to be a dual-monitor setup or single screen.\n"
        "7. Include visible text that clarifies the activity, such as an email subject, PR title, or error.\n"
        "8. Be specific and factual; avoid assumptions about intent.\n\n"
        "Write 2-4 complete sentences. End the final sentence with punctuation. "
        "Return only the final activity log entry; do not return hidden reasoning or a partial fragment.\n\n"
        "If the screen is mostly idle, such as desktop, wallpaper, or lock screen, state that clearly.\n\n"
        f"Additional metadata sent with the screenshot:\n{content_text or '(none)'}"
    )

    try:
        return await _call_vision_for_activity_log(image_path, prompt)
    except AmbientAnalysisIncompleteError:
        retry_prompt = (
            f"{prompt}\n\n"
            "The previous response was incomplete. Retry with a complete, self-contained "
            "2-4 sentence activity log entry ending in punctuation."
        )
        return await _call_vision_for_activity_log(image_path, retry_prompt)


async def _analyze_image_payload(image: Dict[str, Any]) -> str:
    temp_path = None
    try:
        suffix = ".jpg"
        if "png" in image["image_mime"]:
            suffix = ".png"
        elif "webp" in image["image_mime"]:
            suffix = ".webp"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(base64.b64decode(image["image_data"], validate=True))
            temp_path = Path(temp_file.name)

        return await _analyze_screenshot(
            temp_path,
            image["timestamp"],
            image["content_text"],
        )
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


async def analyze_ambient_ingest_content(
    *,
    session_id: str,
    role: str,
    content: Any,
    timestamp: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Analyze screenshots in an ingest message before any durable write.

    Raises when vision analysis fails so callers can reject the ingest instead
    of storing partial or unanalyzed ambient events.
    """
    images = _extract_images_from_content(
        message_id=0,
        role=role,
        content=content,
        timestamp=timestamp if timestamp is not None else datetime.now(timezone.utc).timestamp(),
        reject_unsupported_images=True,
    )
    analyses = []
    for image in images:
        analysis = await _analyze_image_payload(image)
        analyses.append(
            {
                "image_hash": image["image_hash"],
                "image_index": image.get("image_index", 0),
                "session_id": session_id,
                "timestamp": image["timestamp"],
                "role": image.get("role"),
                "content_text": image.get("content_text", ""),
                "analysis": analysis,
            }
        )

    return analyses


async def read_ambient_context_tool(
    limit: int = 10,
    session_id: str = AMBIENT_DEFAULT_SESSION_ID,
    start_time: Any = None,
    end_time: Any = None,
) -> str:
    """Read persisted ambient screenshot analyses without touching image data."""
    from tools.interrupt import is_interrupted

    if is_interrupted():
        return tool_error("Interrupted", success=False)

    try:
        start_timestamp = _parse_timestamp_bound(start_time, "start_time")
        end_timestamp = _parse_timestamp_bound(end_time, "end_time")
    except ValueError as exc:
        return tool_error(str(exc), success=False)

    if (
        start_timestamp is not None
        and end_timestamp is not None
        and start_timestamp > end_timestamp
    ):
        return tool_error("start_time must be less than or equal to end_time", success=False)

    results = _get_ambient_analysis_events(
        limit=limit,
        session_id=session_id,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
    )
    if not results:
        return json.dumps(
            {
                "success": True,
                "session_id": session_id,
                "count": 0,
                "images": [],
                "latest_analysis": None,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
            },
            indent=2,
            ensure_ascii=False,
        )

    results.sort(key=lambda item: item["timestamp"], reverse=True)

    return json.dumps(
        {
            "success": True,
            "session_id": session_id,
            "count": len(results),
            "images": results,
            "latest_analysis": results[0]["analysis"] if results else None,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
        },
        indent=2,
        ensure_ascii=False,
    )


READ_AMBIENT_CONTEXT_SCHEMA = {
    "name": "read_ambient_context",
    "description": (
        "Read persisted ambient screenshot descriptions from the background ingest session. "
        "Screenshots are analyzed when they are received, and the source image is discarded; "
        "this tool only retrieves stored descriptions. Use this to understand what the user "
        "was doing at specific points in time."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of recent ambient descriptions to return (default 10).",
                "default": 10,
            },
            "session_id": {
                "type": "string",
                "description": "The ambient ingest session ID to read from (default ambient:journal:context).",
                "default": AMBIENT_DEFAULT_SESSION_ID,
            },
            "start_time": {
                "anyOf": [{"type": "number"}, {"type": "string"}],
                "description": (
                    "Optional inclusive lower timestamp bound. Use Unix epoch seconds, or a "
                    "timezone-aware ISO-8601/RFC3339 string such as 2026-05-07T09:30:00Z."
                ),
            },
            "end_time": {
                "anyOf": [{"type": "number"}, {"type": "string"}],
                "description": (
                    "Optional inclusive upper timestamp bound. Use Unix epoch seconds, or a "
                    "timezone-aware ISO-8601/RFC3339 string such as 2026-05-07T10:30:00+05:30."
                ),
            },
        },
        "required": [],
    },
}


def _handle_read_ambient_context(args: Dict[str, Any], **kw: Any) -> Awaitable[str]:
    return read_ambient_context_tool(
        limit=args.get("limit", 10),
        session_id=args.get("session_id", AMBIENT_DEFAULT_SESSION_ID),
        start_time=args.get("start_time"),
        end_time=args.get("end_time"),
    )


def check_ambient_context_requirements() -> bool:
    return True


registry.register(
    name="read_ambient_context",
    toolset="ambient",
    schema=READ_AMBIENT_CONTEXT_SCHEMA,
    handler=_handle_read_ambient_context,
    check_fn=check_ambient_context_requirements,
    is_async=True,
    emoji="📸",
)
