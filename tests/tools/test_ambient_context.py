import base64
import hashlib
import json

import pytest

from hermes_constants import get_hermes_home
from tools.ambient_context import (
    AmbientAnalysisIncompleteError,
    AmbientIngestValidationError,
    AMBIENT_DEFAULT_SESSION_ID,
    _CONTENT_JSON_PREFIX,
    _analyze_screenshot,
    _ambient_analysis_looks_incomplete,
    _extract_images_from_content,
    _normalize_ambient_activity_log,
    read_ambient_context_tool,
    store_ambient_ingest_events,
)


def _data_url(payload: bytes = b"screenshot") -> str:
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def test_extract_images_decodes_sessiondb_json_prefix():
    payload = b"image bytes"
    content = _CONTENT_JSON_PREFIX + json.dumps(
        [
            {"type": "text", "text": "metadata"},
            {"type": "image_url", "image_url": {"url": _data_url(payload)}},
        ]
    )

    images = _extract_images_from_content(
        message_id=123,
        role="user",
        content=content,
        timestamp=10.0,
    )

    assert len(images) == 1
    assert images[0]["message_id"] == 123
    assert images[0]["image_index"] == 0
    assert images[0]["content_text"] == "metadata"
    assert images[0]["image_hash"] == hashlib.sha256(payload).hexdigest()


def test_extract_images_rejects_remote_url_for_ingest():
    with pytest.raises(AmbientIngestValidationError, match="inline data:image"):
        _extract_images_from_content(
            message_id=1,
            role="user",
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/screenshot.png"},
                }
            ],
            timestamp=1.0,
            reject_unsupported_images=True,
        )


def test_extract_images_rejects_invalid_base64_for_ingest():
    with pytest.raises(AmbientIngestValidationError, match="Failed to parse"):
        _extract_images_from_content(
            message_id=1,
            role="user",
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,not base64!"},
                }
            ],
            timestamp=1.0,
            reject_unsupported_images=True,
        )


def test_ambient_analysis_incomplete_detector():
    assert _ambient_analysis_looks_incomplete("Provider integrations")
    assert _ambient_analysis_looks_incomplete("Short.")
    assert not _ambient_analysis_looks_incomplete(
        "Zen Browser is open on the Hermes dashboard. The page shows provider integrations."
    )


def test_normalize_ambient_activity_log_preserves_long_fragment():
    analysis = (
        "Zen Browser is open on the Hermes dashboard. "
        "The page shows provider integrations. "
        "A terminal is visible in the second monitor. "
        "The user appears to be reviewing configuration state. "
        "Extra sentence that should not be stored. "
        + "verbose details " * 200
        + "cut off in the middle of"
    )

    normalized, reason = _normalize_ambient_activity_log(analysis)

    assert reason is None
    assert normalized == analysis


@pytest.mark.asyncio
async def test_read_ambient_context_reads_stored_descriptions_without_state_db():
    store_ambient_ingest_events(
        session_id=AMBIENT_DEFAULT_SESSION_ID,
        analyses=[
            {
                "image_hash": "hash-1",
                "image_index": 0,
                "session_id": AMBIENT_DEFAULT_SESSION_ID,
                "timestamp": 10.0,
                "role": "user",
                "content_text": "metadata",
                "analysis": "The user is working in a terminal.",
            }
        ],
    )

    result = json.loads(await read_ambient_context_tool())

    assert result["count"] == 1
    assert result["latest_analysis"] == "The user is working in a terminal."
    assert result["images"][0]["event_id"] == result["images"][0]["message_id"]
    assert result["images"][0]["content_text"] == "metadata"
    assert "cached" not in result["images"][0]


@pytest.mark.asyncio
async def test_read_ambient_context_does_not_reanalyze(monkeypatch):
    async def _fail_if_called(**kwargs):
        raise AssertionError("read_ambient_context must not call vision")

    monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", _fail_if_called)
    store_ambient_ingest_events(
        session_id=AMBIENT_DEFAULT_SESSION_ID,
        analyses=[
            {
                "image_hash": "hash-2",
                "image_index": 0,
                "session_id": AMBIENT_DEFAULT_SESSION_ID,
                "timestamp": 11.0,
                "role": "user",
                "content_text": "",
                "analysis": "The user is reading a dashboard.",
            }
        ],
    )

    result = json.loads(await read_ambient_context_tool())

    assert result["count"] == 1
    assert result["latest_analysis"] == "The user is reading a dashboard."
    assert "skip_cached_ignored" not in result


@pytest.mark.asyncio
async def test_analyze_screenshot_rejects_vision_failure(monkeypatch, tmp_path):
    async def _fake_vision_analyze_tool(**kwargs):
        return json.dumps({"success": False, "error": "vision returned no final content"})

    monkeypatch.setattr(
        "tools.vision_tools.vision_analyze_tool",
        _fake_vision_analyze_tool,
    )
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"not validated by patched vision tool")

    with pytest.raises(RuntimeError, match="no final content"):
        await _analyze_screenshot(image_path, 1.0, "")


@pytest.mark.asyncio
async def test_analyze_screenshot_requires_non_empty_final_analysis(monkeypatch, tmp_path):
    async def _fake_vision_analyze_tool(**kwargs):
        return json.dumps({"success": True, "analysis": "   "})

    monkeypatch.setattr(
        "tools.vision_tools.vision_analyze_tool",
        _fake_vision_analyze_tool,
    )
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"not validated by patched vision tool")

    with pytest.raises(RuntimeError, match="no final content"):
        await _analyze_screenshot(image_path, 1.0, "")


@pytest.mark.asyncio
async def test_analyze_screenshot_retries_incomplete_final_analysis(monkeypatch, tmp_path):
    responses = iter(
        [
            {"success": True, "analysis": "Provider integrations"},
            {"success": True, "analysis": "Zen Browser is open displaying the Hermes dashboard. The page shows provider integrations and configuration controls."},
        ]
    )
    prompts = []

    async def _fake_vision_analyze_tool(**kwargs):
        prompts.append(kwargs["user_prompt"])
        return json.dumps(next(responses))

    monkeypatch.setattr(
        "tools.vision_tools.vision_analyze_tool",
        _fake_vision_analyze_tool,
    )
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"not validated by patched vision tool")

    analysis = await _analyze_screenshot(image_path, 1.0, "")

    assert analysis.endswith("controls.")
    assert len(prompts) == 2
    assert "previous response was incomplete" in prompts[1]


@pytest.mark.asyncio
async def test_analyze_screenshot_accepts_long_truncated_analysis(monkeypatch, tmp_path):
    long_analysis = (
        "Zen Browser is open on the Hermes dashboard. "
        "The page shows provider integrations. "
        "A terminal is visible in the second monitor. "
        "The user appears to be reviewing configuration state. "
        "Extra sentence that should not be stored. "
        + "verbose details " * 500
        + "cut off in the middle of"
    )
    prompts = []

    async def _fake_vision_analyze_tool(**kwargs):
        prompts.append(kwargs["user_prompt"])
        assert "max_tokens" not in kwargs
        return json.dumps(
            {
                "success": True,
                "analysis": long_analysis,
                "metadata": {
                    "finish_reason": "length",
                    "configured_provider": "opencode-go",
                    "response_model": "kimi-k2.6",
                    "max_tokens": 2000,
                },
            }
        )

    monkeypatch.setattr(
        "tools.vision_tools.vision_analyze_tool",
        _fake_vision_analyze_tool,
    )
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"not validated by patched vision tool")

    analysis = await _analyze_screenshot(image_path, 1.0, "")

    assert analysis == long_analysis
    assert len(prompts) == 1
    capture_path = get_hermes_home() / "logs" / "ambient_vision_captures.jsonl"
    records = [json.loads(line) for line in capture_path.read_text().splitlines()]
    assert records[-1]["event"] == "accepted_shape_warning"
    assert records[-1]["reason"] == "missing_terminal_punctuation"
    assert records[-1]["finish_reason"] == "length"
    assert records[-1]["configured_provider"] == "opencode-go"
    assert "cut off in the middle of" in records[-1]["analysis_tail"]


@pytest.mark.asyncio
async def test_analyze_screenshot_rejects_repeated_incomplete_analysis(monkeypatch, tmp_path):
    async def _fake_vision_analyze_tool(**kwargs):
        return json.dumps({"success": True, "analysis": "The page shows Provider integrations"})

    monkeypatch.setattr(
        "tools.vision_tools.vision_analyze_tool",
        _fake_vision_analyze_tool,
    )
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"not validated by patched vision tool")

    with pytest.raises(AmbientAnalysisIncompleteError):
        await _analyze_screenshot(image_path, 1.0, "")
