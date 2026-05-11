import base64
import hashlib
import json

import pytest

from tools.ambient_context import (
    AmbientIngestValidationError,
    _CONTENT_JSON_PREFIX,
    _analyze_screenshot,
    _extract_images_from_content,
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
