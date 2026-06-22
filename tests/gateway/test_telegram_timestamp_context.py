"""Tests for Telegram per-chat timestamp context injection and replay."""

import sys
import threading
import types
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


class _CapturingAgent:
    last_init = None
    last_run = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.model = "gpt-5.4"
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0

    def run_conversation(
        self,
        user_message,
        conversation_history=None,
        task_id=None,
        persist_user_message=None,
        persist_user_sent_at=None,
    ):
        type(self).last_run = {
            "user_message": user_message,
            "conversation_history": conversation_history,
            "task_id": task_id,
            "persist_user_message": persist_user_message,
            "persist_user_sent_at": persist_user_sent_at,
        }
        return {
            "final_response": "ok",
            "messages": [
                {"role": "user", "content": persist_user_message or user_message},
                {"role": "assistant", "content": "ok"},
            ],
            "api_calls": 1,
            "completed": True,
        }


def _install_fake_agent(monkeypatch):
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = "Global prompt"
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._service_tier = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._smart_model_routing = {}
    runner._running_agents = {}
    runner._pending_model_notes = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(streaming=None, thread_sessions_per_user=False)
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(session_id="session-1"),
        load_transcript=lambda session_id: [],
    )
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._enrich_message_with_vision = AsyncMock(return_value="ENRICHED")
    runner._enrich_message_with_transcription = AsyncMock(return_value="TRANSCRIBED")
    runner._has_setup_skill = lambda: False
    runner._model = "gpt-5.4"
    runner._base_url = None
    return runner


def _make_source(
    chat_id="-1003961464202",
    thread_id=None,
    platform=Platform.TELEGRAM,
    chat_type="group",
    parent_chat_id=None,
) -> SessionSource:
    return SessionSource(
        platform=platform,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id="806682516",
        user_name="Aditya",
        thread_id=thread_id,
        parent_chat_id=parent_chat_id,
    )


def test_normalize_sent_at_attaches_configured_timezone_to_naive_string():
    assert (
        gateway_run._normalize_sent_at(
            "2026-05-21T15:51:33",
            user_config={"timezone": "Asia/Kolkata"},
        )
        == "2026-05-21T15:51:33+05:30"
    )


@pytest.mark.asyncio
async def test_prepare_inbound_message_text_injects_timestamp_for_enabled_chat(monkeypatch):
    runner = _make_runner()
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "telegram": {"timestamp_context_chats": {"-1003961464202": True}},
            "timezone": "Asia/Kolkata",
        },
    )

    event = MessageEvent(
        text="today was messy",
        source=_make_source(),
        timestamp=datetime(2026, 4, 24, 11, 40, 12, tzinfo=timezone.utc),
    )

    message = await runner._prepare_inbound_message_text(
        event=event,
        source=event.source,
        history=[],
    )

    assert message == "[Sent at: 2026-04-24T17:10:12+05:30]\n\ntoday was messy"


@pytest.mark.asyncio
async def test_prepare_inbound_message_text_skips_timestamp_for_other_chat(monkeypatch):
    runner = _make_runner()
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"telegram": {"timestamp_context_chats": {"-111": True}}},
    )

    event = MessageEvent(
        text="today was messy",
        source=_make_source(),
        timestamp=datetime(2026, 4, 24, 11, 40, 12, tzinfo=timezone.utc),
    )

    message = await runner._prepare_inbound_message_text(
        event=event,
        source=event.source,
        history=[],
    )

    assert message == "today was messy"


@pytest.mark.asyncio
async def test_prepare_inbound_message_text_skips_timestamp_for_false_string_flag(monkeypatch):
    runner = _make_runner()
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"telegram": {"timestamp_context_chats": {"-1003961464202": "false"}}},
    )

    event = MessageEvent(
        text="today was messy",
        source=_make_source(),
        timestamp=datetime(2026, 4, 24, 11, 40, 12, tzinfo=timezone.utc),
    )

    message = await runner._prepare_inbound_message_text(
        event=event,
        source=event.source,
        history=[],
    )

    assert message == "today was messy"


@pytest.mark.asyncio
async def test_prepare_inbound_message_text_injects_timestamp_for_enabled_discord_channel(monkeypatch):
    runner = _make_runner()
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "discord": {"timestamp_context_chats": {"1234567890": True}},
            "timezone": "Asia/Kolkata",
        },
    )

    event = MessageEvent(
        text="today was messy",
        source=_make_source(chat_id="1234567890", platform=Platform.DISCORD),
        timestamp=datetime(2026, 4, 24, 11, 40, 12, tzinfo=timezone.utc),
    )

    message = await runner._prepare_inbound_message_text(
        event=event,
        source=event.source,
        history=[],
    )

    assert message == "[Sent at: 2026-04-24T17:10:12+05:30]\n\ntoday was messy"


@pytest.mark.asyncio
async def test_prepare_inbound_message_text_injects_timestamp_for_discord_thread_parent(monkeypatch):
    runner = _make_runner()
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "discord": {"timestamp_context_chats": {"1234567890": True}},
            "timezone": "Asia/Kolkata",
        },
    )

    event = MessageEvent(
        text="today was messy",
        source=_make_source(
            chat_id="2222222222",
            thread_id="2222222222",
            parent_chat_id="1234567890",
            platform=Platform.DISCORD,
            chat_type="thread",
        ),
        timestamp=datetime(2026, 4, 24, 11, 40, 12, tzinfo=timezone.utc),
    )

    message = await runner._prepare_inbound_message_text(
        event=event,
        source=event.source,
        history=[],
    )

    assert message == "[Sent at: 2026-04-24T17:10:12+05:30]\n\n[Aditya] today was messy"


@pytest.mark.parametrize(
    ("source", "platform_key", "enabled_id", "session_key"),
    [
        (
            _make_source(),
            "telegram",
            "-1003961464202",
            "telegram:group:-1003961464202",
        ),
        (
            _make_source(chat_id="1234567890", platform=Platform.DISCORD),
            "discord",
            "1234567890",
            "discord:group:1234567890",
        ),
    ],
)
@pytest.mark.asyncio
async def test_run_agent_replays_sent_at_history_and_persists_clean_user_message(
    monkeypatch,
    tmp_path,
    source,
    platform_key,
    enabled_id,
    session_key,
):
    _install_fake_agent(monkeypatch)
    runner = _make_runner()

    (tmp_path / "config.yaml").write_text("agent:\n  system_prompt: Global prompt\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            platform_key: {"timestamp_context_chats": {enabled_id: True}},
            "display": {},
            "agent": {"system_prompt": "Global prompt"},
            "timezone": "Asia/Kolkata",
        },
    )
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "***",
        },
    )

    import hermes_cli.tools_config as tools_config

    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

    _CapturingAgent.last_run = None
    result = await runner._run_agent(
        message="[Sent at: 2026-04-24T17:10:12+05:30]\n\ntoday was messy",
        context_prompt="Context prompt",
        history=[
            {
                "role": "user",
                "content": "earlier note",
                "sent_at": "2026-04-23T03:45:00+00:00",
            }
        ],
        source=source,
        session_id="session-1",
        session_key=session_key,
        channel_prompt=None,
        persist_user_message="today was messy",
        persist_user_sent_at="2026-04-24T17:10:12+05:30",
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_run["user_message"] == "[Sent at: 2026-04-24T17:10:12+05:30]\n\ntoday was messy"
    assert _CapturingAgent.last_run["persist_user_message"] == "today was messy"
    assert _CapturingAgent.last_run["persist_user_sent_at"] == "2026-04-24T17:10:12+05:30"
    assert _CapturingAgent.last_run["conversation_history"] == [
        {
            "role": "user",
            "content": "[Sent at: 2026-04-23T09:15:00+05:30]\n\nearlier note",
        }
    ]
