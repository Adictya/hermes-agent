"""End-to-end tests for inline image inputs on /v1/chat/completions and /v1/responses.

Covers the multimodal normalization path added to the API server.  Unlike the
adapter-level tests that patch ``_run_agent``, these tests patch
``AIAgent.run_conversation`` instead so the adapter's full request-handling
path (including the ``run_agent`` prologue that used to crash on list content)
executes against a real aiohttp app.
"""

import base64
import json
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _content_has_visible_payload,
    _normalize_multimodal_content,
    cors_middleware,
    security_headers_middleware,
)
from hermes_constants import get_hermes_home
from tools.ambient_context import AMBIENT_DEFAULT_SESSION_ID, read_ambient_context_tool


# ---------------------------------------------------------------------------
# Pure-function tests for _normalize_multimodal_content
# ---------------------------------------------------------------------------


class TestNormalizeMultimodalContent:
    def test_string_passthrough(self):
        assert _normalize_multimodal_content("hello") == "hello"

    def test_none_returns_empty_string(self):
        assert _normalize_multimodal_content(None) == ""

    def test_text_only_list_collapses_to_string(self):
        content = [{"type": "text", "text": "hi"}, {"type": "text", "text": "there"}]
        assert _normalize_multimodal_content(content) == "hi\nthere"

    def test_responses_input_text_canonicalized(self):
        content = [{"type": "input_text", "text": "hello"}]
        assert _normalize_multimodal_content(content) == "hello"

    def test_image_url_preserved_with_text(self):
        content = [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "high"}},
        ]
        out = _normalize_multimodal_content(content)
        assert isinstance(out, list)
        assert out == [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "high"}},
        ]

    def test_input_image_converted_to_canonical_shape(self):
        content = [
            {"type": "input_text", "text": "hi"},
            {"type": "input_image", "image_url": "https://example.com/cat.png"},
        ]
        out = _normalize_multimodal_content(content)
        assert out == [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ]

    def test_data_image_url_accepted(self):
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
        out = _normalize_multimodal_content(content)
        assert out == [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]

    def test_non_image_data_url_rejected(self):
        content = [{"type": "image_url", "image_url": {"url": "data:text/plain;base64,SGVsbG8="}}]
        with pytest.raises(ValueError) as exc:
            _normalize_multimodal_content(content)
        assert str(exc.value).startswith("unsupported_content_type:")

    def test_file_part_rejected(self):
        with pytest.raises(ValueError) as exc:
            _normalize_multimodal_content([{"type": "file", "file": {"file_id": "f_1"}}])
        assert str(exc.value).startswith("unsupported_content_type:")

    def test_input_file_part_rejected(self):
        with pytest.raises(ValueError) as exc:
            _normalize_multimodal_content([{"type": "input_file", "file_id": "f_1"}])
        assert str(exc.value).startswith("unsupported_content_type:")

    def test_missing_url_rejected(self):
        with pytest.raises(ValueError) as exc:
            _normalize_multimodal_content([{"type": "image_url", "image_url": {}}])
        assert str(exc.value).startswith("invalid_image_url:")

    def test_bad_scheme_rejected(self):
        with pytest.raises(ValueError) as exc:
            _normalize_multimodal_content([{"type": "image_url", "image_url": {"url": "ftp://example.com/x.png"}}])
        assert str(exc.value).startswith("invalid_image_url:")

    def test_unknown_part_type_rejected(self):
        with pytest.raises(ValueError) as exc:
            _normalize_multimodal_content([{"type": "audio", "audio": {}}])
        assert str(exc.value).startswith("unsupported_content_type:")


class TestContentHasVisiblePayload:
    def test_non_empty_string(self):
        assert _content_has_visible_payload("hello")

    def test_whitespace_only_string(self):
        assert not _content_has_visible_payload("   ")

    def test_list_with_image_only(self):
        assert _content_has_visible_payload([{"type": "image_url", "image_url": {"url": "x"}}])

    def test_list_with_only_empty_text(self):
        assert not _content_has_visible_payload([{"type": "text", "text": ""}])


# ---------------------------------------------------------------------------
# HTTP integration — real aiohttp client hitting the adapter handlers
# ---------------------------------------------------------------------------


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_get("/v1/responses/{response_id}", adapter._handle_get_response)
    return app


def _inline_data_image(payload: bytes = b"fake screenshot bytes") -> str:
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


@pytest.fixture
def adapter():
    return _make_adapter()


class TestChatCompletionsMultimodalHTTP:
    @pytest.mark.asyncio
    async def test_inline_image_preserved_to_run_agent(self, adapter):
        """Multimodal user content reaches _run_agent as a list of parts."""
        image_payload = [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "high"}},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter,
                "_run_agent",
                new=MagicMock(),
            ) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "A cat.", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                mock_run.side_effect = _stub

                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": image_payload}],
                    },
                )

            assert resp.status == 200, await resp.text()
            assert mock_run.captured["user_message"] == image_payload

    @pytest.mark.asyncio
    async def test_text_only_array_collapses_to_string(self, adapter):
        """Text-only array becomes a plain string so logging stays unchanged."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "ok", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                mock_run.side_effect = _stub

                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                        ],
                    },
                )

            assert resp.status == 200, await resp.text()
            assert mock_run.captured["user_message"] == "hello"

    @pytest.mark.asyncio
    async def test_file_part_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={
                    "model": "hermes-agent",
                    "messages": [
                        {"role": "user", "content": [{"type": "file", "file": {"file_id": "f_1"}}]},
                    ],
                },
            )
            assert resp.status == 400
            body = await resp.json()
        assert body["error"]["code"] == "unsupported_content_type"
        assert body["error"]["param"] == "messages[0].content"


class TestChatCompletionsAmbientIngestHTTP:
    @pytest.mark.asyncio
    async def test_ingest_defaults_to_ambient_session_and_caches_analysis(self, adapter, monkeypatch):
        async def _fake_vision_analyze_tool(**kwargs):
            return json.dumps({"success": True, "analysis": "Aditya is using Terminal to review the ambient screenshot pipeline."})

        monkeypatch.setattr(
            "tools.vision_tools.vision_analyze_tool",
            _fake_vision_analyze_tool,
        )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                headers={"X-Hermes-Mode": "ingest"},
                json={
                    "model": "hermes-agent",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "screenshot"},
                                {"type": "image_url", "image_url": {"url": _inline_data_image()}},
                            ],
                        }
                    ],
                },
            )

        assert resp.status == 204, await resp.text()
        assert resp.headers["X-Hermes-Session-Id"] == AMBIENT_DEFAULT_SESSION_ID

        ambient = json.loads(await read_ambient_context_tool())
        assert ambient["count"] == 1
        assert ambient["latest_analysis"] == "Aditya is using Terminal to review the ambient screenshot pipeline."
        assert "cached" not in ambient["images"][0]
        assert not (get_hermes_home() / "state.db").exists()

    @pytest.mark.asyncio
    async def test_ingest_vision_failure_does_not_save_message(self, adapter, monkeypatch):
        async def _fake_vision_analyze_tool(**kwargs):
            return json.dumps({"success": False, "error": "no final content"})

        monkeypatch.setattr(
            "tools.vision_tools.vision_analyze_tool",
            _fake_vision_analyze_tool,
        )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                headers={"X-Hermes-Mode": "ingest"},
                json={
                    "model": "hermes-agent",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": _inline_data_image()}},
                            ],
                        }
                    ],
                },
            )

        assert resp.status == 502
        ambient = json.loads(await read_ambient_context_tool())
        assert ambient["count"] == 0

    @pytest.mark.asyncio
    async def test_ingest_cache_failure_does_not_store_event(self, adapter, monkeypatch):
        async def _fake_vision_analyze_tool(**kwargs):
            return json.dumps({"success": True, "analysis": "The screenshot shows a terminal workflow with visible analysis content."})

        def _fail_store(**kwargs):
            raise RuntimeError("cache unavailable")

        monkeypatch.setattr(
            "tools.vision_tools.vision_analyze_tool",
            _fake_vision_analyze_tool,
        )
        monkeypatch.setattr("tools.ambient_context.store_ambient_ingest_events", _fail_store)

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                headers={"X-Hermes-Mode": "ingest"},
                json={
                    "model": "hermes-agent",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": _inline_data_image()}},
                            ],
                        }
                    ],
                },
            )

        assert resp.status == 500
        ambient = json.loads(await read_ambient_context_tool())
        assert ambient["count"] == 0

    @pytest.mark.asyncio
    async def test_ingest_rejects_remote_image_urls(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                headers={"X-Hermes-Mode": "ingest"},
                json={
                    "model": "hermes-agent",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "https://example.com/screenshot.png"},
                                },
                            ],
                        }
                    ],
                },
            )

        assert resp.status == 400
        body = await resp.json()
        assert body["error"]["code"] == "invalid_ambient_ingest"
        assert body["error"]["param"] == "messages[0].content"

    @pytest.mark.asyncio
    async def test_non_image_data_url_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={
                    "model": "hermes-agent",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "data:text/plain;base64,SGVsbG8="},
                                },
                            ],
                        },
                    ],
                },
            )
            assert resp.status == 400
            body = await resp.json()
        assert body["error"]["code"] == "unsupported_content_type"


class TestResponsesMultimodalHTTP:
    @pytest.mark.asyncio
    async def test_responses_ingest_uses_same_ambient_pipeline(self, adapter, monkeypatch):
        async def _fake_vision_analyze_tool(**kwargs):
            return json.dumps({"success": True, "analysis": "Responses ingest stores a terminal screenshot description without retaining the image."})

        monkeypatch.setattr(
            "tools.vision_tools.vision_analyze_tool",
            _fake_vision_analyze_tool,
        )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                headers={"X-Hermes-Mode": "ingest"},
                json={
                    "model": "hermes-agent",
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "Describe."},
                                {"type": "input_image", "image_url": _inline_data_image()},
                            ],
                        }
                    ],
                },
            )

        assert resp.status == 204, await resp.text()
        ambient = json.loads(await read_ambient_context_tool())
        assert ambient["latest_analysis"] == "Responses ingest stores a terminal screenshot description without retaining the image."

    @pytest.mark.asyncio
    async def test_input_image_canonicalized_and_forwarded(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "ok", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                mock_run.side_effect = _stub

                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": "Describe."},
                                    {
                                        "type": "input_image",
                                        "image_url": "https://example.com/cat.png",
                                    },
                                ],
                            }
                        ],
                    },
                )

            assert resp.status == 200, await resp.text()
            expected = [
                {"type": "text", "text": "Describe."},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            ]
            assert mock_run.captured["user_message"] == expected

    @pytest.mark.asyncio
    async def test_input_file_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                json={
                    "model": "hermes-agent",
                    "input": [
                        {
                            "role": "user",
                            "content": [{"type": "input_file", "file_id": "f_1"}],
                        }
                    ],
                },
            )
            assert resp.status == 400
            body = await resp.json()
        assert body["error"]["code"] == "unsupported_content_type"
