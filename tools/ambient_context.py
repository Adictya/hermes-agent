#!/usr/bin/env python3
"""
Ambient context tool.

Reads screenshots from the ambient ingest session, analyzes them with vision AI,
caches results so each image is only analyzed once, and returns activity
descriptions.
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


def _state_db_path() -> Path:
    return get_hermes_home() / "state.db"


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
        CREATE TABLE IF NOT EXISTS image_analyses (
            image_hash TEXT PRIMARY KEY,
            message_id INTEGER,
            session_id TEXT,
            timestamp REAL,
            analysis TEXT NOT NULL,
            analyzed_at REAL DEFAULT (unixepoch())
        )
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


def _get_cached_analysis(image_hash: str) -> Optional[str]:
    db = _cache_db_path()
    if not db.exists():
        return None

    conn = sqlite3.connect(str(db))
    cursor = conn.cursor()
    cursor.execute("SELECT analysis FROM image_analyses WHERE image_hash = ?", (image_hash,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def _store_analysis(
    image_hash: str,
    message_id: int,
    session_id: str,
    timestamp: float,
    analysis: str,
) -> None:
    _init_cache_db()
    conn = sqlite3.connect(str(_cache_db_path()))
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO image_analyses
        (image_hash, message_id, session_id, timestamp, analysis)
        VALUES (?, ?, ?, ?, ?)
        """,
        (image_hash, message_id, session_id, timestamp, analysis),
    )
    conn.commit()
    conn.close()


def _extract_images_from_messages(
    limit: int = 20,
    session_id: str = "ambient:journal:context",
    start_timestamp: Optional[float] = None,
    end_timestamp: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Return screenshot content blocks from the given persisted session."""
    state_db = _state_db_path()
    if not state_db.exists():
        return []

    where_clauses = ["session_id = ?"]
    params: List[Any] = [session_id]
    if start_timestamp is not None:
        where_clauses.append("timestamp >= ?")
        params.append(start_timestamp)
    if end_timestamp is not None:
        where_clauses.append("timestamp <= ?")
        params.append(end_timestamp)
    params.append(limit)

    conn = sqlite3.connect(str(state_db))
    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT id, session_id, role, content, timestamp
        FROM messages
        WHERE {' AND '.join(where_clauses)}
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        params,
    )
    rows = cursor.fetchall()
    conn.close()

    images = []
    for row in rows:
        msg_id, _sess_id, role, content, timestamp = row
        if not content:
            continue

        try:
            raw = content
            if raw.startswith("\x00json:"):
                raw = raw[7:]

            raw_stripped = raw.strip()
            if raw_stripped.startswith("{") and raw_stripped.endswith("]"):
                raw = "[" + raw_stripped

            data = json.loads(raw)
        except json.JSONDecodeError:
            continue

        blocks = data if isinstance(data, list) else [data]
        if isinstance(data, dict) and "content" in data:
            blocks = data["content"]
            if isinstance(blocks, str):
                blocks = [{"type": "text", "text": blocks}]
        elif isinstance(data, dict):
            blocks = [data]

        text_parts = []
        for block in blocks:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type", "")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "image_url":
                url = block.get("image_url", {}).get("url", "")
                if not url.startswith("data:image/"):
                    continue

                try:
                    header, image_data = url.split(",", 1)
                    mime = header.split(";")[0].split(":")[1]
                    image_hash = hashlib.sha256(image_data.encode()).hexdigest()
                    images.append(
                        {
                            "message_id": msg_id,
                            "timestamp": timestamp,
                            "role": role,
                            "content_text": "\n".join(text_parts).strip(),
                            "image_data": image_data,
                            "image_mime": mime,
                            "image_hash": image_hash,
                        }
                    )
                except Exception:
                    logger.warning("Failed to parse data URL in message %s", msg_id)

    return images


async def _analyze_screenshot(
    image_path: Path,
    timestamp: float,
    content_text: str,
) -> str:
    """Call the vision model with a prompt tuned for screenshot activity logging."""
    from tools.vision_tools import vision_analyze_tool

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
        "If the screen is mostly idle, such as desktop, wallpaper, or lock screen, state that clearly.\n\n"
        f"Additional metadata sent with the screenshot:\n{content_text or '(none)'}"
    )

    result_json = await vision_analyze_tool(
        image_url=str(image_path),
        user_prompt=prompt,
    )
    try:
        result = json.loads(result_json)
        return result.get("analysis", result_json)
    except json.JSONDecodeError:
        return result_json


async def read_ambient_context_tool(
    limit: int = 10,
    session_id: str = "ambient:journal:context",
    skip_cached: bool = False,
    start_time: Any = None,
    end_time: Any = None,
) -> str:
    """Read ambient screenshots from the ingest session and analyze uncached ones."""
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

    _init_cache_db()

    images = _extract_images_from_messages(
        limit=limit,
        session_id=session_id,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
    )
    if not images:
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

    results = []
    for image in images:
        cached = _get_cached_analysis(image["image_hash"])
        if cached and not skip_cached:
            results.append(
                {
                    "message_id": image["message_id"],
                    "timestamp": image["timestamp"],
                    "analysis": cached,
                    "cached": True,
                }
            )
            continue

        temp_path = None
        try:
            suffix = ".jpg"
            if "png" in image["image_mime"]:
                suffix = ".png"
            elif "webp" in image["image_mime"]:
                suffix = ".webp"

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_file.write(base64.b64decode(image["image_data"]))
                temp_path = Path(temp_file.name)

            analysis = await _analyze_screenshot(
                temp_path,
                image["timestamp"],
                image["content_text"],
            )

            try:
                vision_result = json.loads(analysis) if analysis.strip().startswith("{") else None
            except Exception:
                vision_result = None

            if not (vision_result and not vision_result.get("success", True)):
                _store_analysis(
                    image["image_hash"],
                    image["message_id"],
                    session_id,
                    image["timestamp"],
                    analysis,
                )

            results.append(
                {
                    "message_id": image["message_id"],
                    "timestamp": image["timestamp"],
                    "analysis": analysis,
                    "cached": False,
                }
            )
        except Exception as exc:
            logger.error("Failed to analyze screenshot msg=%s: %s", image["message_id"], exc)
            results.append(
                {
                    "message_id": image["message_id"],
                    "timestamp": image["timestamp"],
                    "analysis": f"Error: {exc}",
                    "cached": False,
                    "error": True,
                }
            )
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

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
        "Read and analyze ambient screenshots from the background ingest session. "
        "Returns activity descriptions for each screenshot. Images are analyzed "
        "once and cached; subsequent calls return cached results unless skip_cached is set. "
        "Use this to understand what the user was doing at specific points in time."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of recent messages to scan for screenshots (default 10).",
                "default": 10,
            },
            "session_id": {
                "type": "string",
                "description": "The ambient ingest session ID to read from (default ambient:journal:context).",
                "default": "ambient:journal:context",
            },
            "skip_cached": {
                "type": "boolean",
                "description": "If true, re-analyze images even if a cached analysis exists.",
                "default": False,
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
        session_id=args.get("session_id", "ambient:journal:context"),
        skip_cached=args.get("skip_cached", False),
        start_time=args.get("start_time"),
        end_time=args.get("end_time"),
    )


def check_ambient_context_requirements() -> bool:
    return _state_db_path().exists()


registry.register(
    name="read_ambient_context",
    toolset="ambient",
    schema=READ_AMBIENT_CONTEXT_SCHEMA,
    handler=_handle_read_ambient_context,
    check_fn=check_ambient_context_requirements,
    is_async=True,
    emoji="📸",
)
