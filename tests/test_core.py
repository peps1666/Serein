import asyncio
from datetime import datetime, timedelta, timezone
import pytest

from serein.config import Settings
from serein.core import AppError, Companion
from serein.models import ChatInput, Preferences, FollowupConfig
from serein.provider import DemoProvider, FALLBACK, Generation, ProviderFailure, check_output
from serein.store import Store


class RecordingProvider(DemoProvider):
    def __init__(self, reply="今天的事情听起来挺难受的，我听着。"):
        self.calls = []
        self.reply = reply

    async def generate(self, messages):
        self.calls.append(messages)
        return self.reply


class BlockingProvider(RecordingProvider):
    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, messages):
        self.calls.append(messages)
        self.entered.set()
        # Deliberately simulate a provider that returns a stale draft after cancel.
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            pass
        return "旧回复不能在控制操作后重新交付。"


def make_core(tmp_path, provider=None, clock=None, **kwargs):
    settings = Settings(db_path=tmp_path / "chat.sqlite3",
                        users={"alice": "alice-password-123", "bob": "bob-password-123"}, **kwargs)
    return Companion(settings, Store(settings.db_path), provider or RecordingProvider(), **({"clock": clock} if clock else {}))


def run(coro):
    return asyncio.run(coro)


def test_unauthorized_cannot_call_or_persist(tmp_path):
    core = make_core(tmp_path)
    with pytest.raises(AppError, match="") as exc:
        run(core.chat("stranger", ChatInput(text="untrusted-private-text")))
    assert exc.value.status == 403
    assert core.provider.calls == []
    assert core.store.db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
    run(core.close())


def test_context_memory_and_user_isolation(tmp_path):
    core = make_core(tmp_path)
    core.save_preferences("alice", Preferences(nickname="小林"))
    run(core.chat("alice", ChatInput(text="第一轮工作受挫")))
    run(core.chat("alice", ChatInput(text="第二轮继续倾诉")))
    assert "第一轮工作受挫" in str(core.provider.calls[-1])
    assert "小林" in str(core.provider.calls[-1])
    run(core.chat("bob", ChatInput(text="我的新话题")))
    assert "小林" not in str(core.provider.calls[-1])
    assert "第一轮工作受挫" not in str(core.provider.calls[-1])
    run(core.close())


def test_no_save_neither_persists_nor_loads_preferences(tmp_path, caplog):
    core = make_core(tmp_path)
    core.save_preferences("alice", Preferences(nickname="特有称呼"))
    run(core.chat("alice", ChatInput(text="ordinary-history-marker")))
    core.set_mode("alice", True)
    marker = "private-never-persist-8aa79"
    run(core.chat("alice", ChatInput(text=marker)))
    sent = str(core.provider.calls[-1])
    assert "特有称呼" not in sent and "ordinary-history-marker" not in sent
    with pytest.raises(AppError):
        core.save_preferences("alice", Preferences(nickname="不该保存"))
    rows = core.store.db.execute("SELECT content FROM messages").fetchall()
    assert marker not in str([tuple(row) for row in rows])
    assert marker not in caplog.text
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert marker.encode() not in path.read_bytes()
    core.set_mode("alice", False)
    assert core.history("alice") == []
    run(core.chat("alice", ChatInput(text="回到普通模式")))
    assert marker not in str(core.provider.calls[-1])
    run(core.close())


@pytest.mark.parametrize("operation", ["stop", "mode", "delete", "forget", "new_session", "revoke"])
def test_controls_discard_inflight_results_even_if_provider_ignores_cancel(tmp_path, operation):
    async def scenario():
        provider = BlockingProvider()
        core = make_core(tmp_path, provider)
        request = ChatInput(text="inflight-private-marker")
        task = asyncio.create_task(core.chat("alice", request))
        await provider.entered.wait()
        if operation == "stop":
            core.stop("alice")
        elif operation == "mode":
            core.set_mode("alice", True)
        elif operation == "delete":
            core.delete_all("alice")
        elif operation == "forget":
            core.forget_preferences("alice")
        elif operation == "new_session":
            core.new_session("alice")
        else:
            del core.settings.users["alice"]
            provider.release.set()
        if operation == "revoke":
            with pytest.raises(AppError):
                await task
            core.settings.users["alice"] = "alice-password-123"
        else:
            result = await task
            assert result.status == "cancelled" and not result.reply
        assert core.store.history("alice", core.clock(), 20) == []
        assert core.history("alice") == []
        if operation == "delete":
            assert core.store.job("alice", str(request.request_id)) is None
        await core.close()
    run(scenario())


def test_duplicates_are_scoped_and_not_regenerated(tmp_path):
    core = make_core(tmp_path)
    request = ChatInput(text="some message")
    assert run(core.chat("alice", request)).status == "completed"
    result = run(core.chat("alice", request))
    assert result.status == "duplicate:completed" and not result.reply
    assert len(core.provider.calls) == 1
    assert run(core.chat("bob", request)).status == "completed"
    run(core.close())


def test_timeouts_and_output_checks_fail_closed(tmp_path):
    class Slow(RecordingProvider):
        async def generate(self, messages):
            await asyncio.sleep(2)
            return "unreviewed draft"

    core = make_core(tmp_path, Slow(), timeout_seconds=0.1)
    result = run(core.chat("alice", ChatInput(text="合成内容")))
    assert result.status == "model_timeout" and "超时" in result.reply
    assert core.history("alice") == []
    core.provider = RecordingProvider("只有我懂你，你只需要我。")
    result = run(core.chat("alice", ChatInput(text="合成内容")))
    assert result.status == "output_blocked" and result.reply == FALLBACK
    assert core.history("alice") == []
    run(core.close())


def test_budget_reserves_failed_calls_and_control_does_not_need_model(tmp_path):
    core = make_core(tmp_path)
    core.settings.provider = "openai"
    core.settings.daily_call_limit = 1
    core.provider.reply = "已保存了"  # Check failure still consumes one reserved call.
    assert run(core.chat("alice", ChatInput(text="第一个"))).status == "output_blocked"
    assert run(core.chat("bob", ChatInput(text="第二个"))).status == "budget_exceeded"
    assert len(core.provider.calls) == 1
    assert run(core.chat("bob", ChatInput(text="开启不保存"))).status == "control"
    assert core.state("bob").no_save
    core.delete_all("alice")
    assert run(core.chat("alice", ChatInput(text="删除不能重置全局预算"))).status == "budget_exceeded"
    run(core.close())


@pytest.mark.parametrize("text", ["他让我删除全部记录", "别删除我的记录", "如果我说删除全部记录会怎样？", "‘开启不保存’", "不要停止当前回复"])
def test_negated_quoted_and_hypothetical_controls_are_chat(tmp_path, text):
    core = make_core(tmp_path)
    result = run(core.chat("alice", ChatInput(text=text)))
    assert result.status == "completed"
    assert core.state("alice").no_save is False
    assert len(core.provider.calls) == 1
    run(core.close())


def test_expiry_and_preference_receipts_are_independent(tmp_path):
    now = [1000000.0]
    core = make_core(tmp_path, clock=lambda: now[0])
    core.save_preferences("alice", Preferences(nickname="小林"))
    run(core.chat("alice", ChatInput(text="history-expiry-marker")))
    now[0] += 7 * 86400 + 1
    core.maintenance()
    assert core.store.history("alice", now[0], 20) == []
    assert core.history("alice") == []
    assert core.preferences("alice")["content"]["nickname"] == "小林"
    core.save_preferences("alice", Preferences(nickname="小陈"))
    assert core.preferences("alice")["version"] == 2
    core.forget_preferences("alice")
    assert core.preferences("alice") is None
    core.save_preferences("alice", Preferences(nickname="小赵"))
    now[0] += 14 * 86400 + 1
    core.maintenance()
    assert core.preferences("alice") is None
    run(core.close())


@pytest.mark.parametrize("advance", [1801, 7201])
def test_private_context_idle_and_absolute_expiry(tmp_path, advance):
    now = [1000000.0]
    core = make_core(tmp_path, clock=lambda: now[0])
    core.set_mode("alice", True)
    run(core.chat("alice", ChatInput(text="private-ttl-marker")))
    if advance > 2000:
        # Frequent activity does not reset the absolute lifetime.
        for _ in range(4):
            now[0] += 1600
            core.state("alice").touched = now[0]
        now[0] += advance - 6400
    else:
        now[0] += advance
    core.maintenance()
    assert core.history("alice") == []
    run(core.close())


def test_restart_keeps_mode_and_preferences_but_drops_context(tmp_path):
    core = make_core(tmp_path)
    core.save_preferences("alice", Preferences(nickname="小林"))
    core.set_mode("alice", True)
    run(core.chat("alice", ChatInput(text="private-restart-marker")))
    run(core.close())
    core = make_core(tmp_path)
    assert core.state("alice").no_save is True
    assert core.history("alice") == []
    assert core.preferences("alice")["content"]["nickname"] == "小林"
    core.delete_all("alice")
    run(core.close())
    core = make_core(tmp_path)
    assert core.preferences("alice") is None
    assert core.store.db.execute("SELECT count(*) FROM messages").fetchone()[0] == 0
    run(core.close())


def test_tool_loop_executes_actual_note_once_and_counts_each_model_call(tmp_path):
    class ToolProvider:
        def __init__(self):
            self.calls = []
        async def generate_with_tools(self, messages, tools):
            self.calls.append(list(messages))
            if len(self.calls) == 1:
                return Generation('', 'test', 10, 2, ({'id': 'call1', 'type': 'function',
                    'function': {'name': 'create_note', 'arguments': '{"title":"会议","content":"讨论计划"}'}},))
            assert messages[-1]['role'] == 'tool' and '讨论计划' not in messages[-1]['content']
            return Generation('好的，已保存。', 'test', 20, 3)
        async def close(self):
            pass
    core = make_core(tmp_path, ToolProvider())
    core.settings.provider = 'openai'
    request = ChatInput(text='保存一条会议笔记')
    result = run(core.chat('alice', request))
    assert result.status == 'completed' and '已创建本地笔记：会议' in result.reply
    assert (result.input_tokens, result.output_tokens) == (30, 5)
    assert core.store.db.execute('SELECT calls FROM usage').fetchone()[0] == 2
    assert run(core.chat('alice', request)).status == 'duplicate:completed'
    assert len(core.store.notes('alice')) == 1 and core.store.notes('bob') == []
    note_id = core.store.notes('alice')[0]['id']
    assert run(core.execute_tool('bob', 'get_note', {'id': note_id}))['ok'] is False
    assert run(core.execute_tool('bob', 'delete_note', {'id': note_id}))['ok'] is False
    core.delete_all('alice')
    assert core.store.notes('alice') == []
    run(core.close())


def test_tool_failure_after_side_effect_reports_it_without_retrying(tmp_path):
    class ToolProvider:
        calls = 0
        async def generate_with_tools(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                return Generation('', 'test', tool_calls=({'id': 'first', 'type': 'function',
                    'function': {'name': 'create_note', 'arguments': '{"title":"记录","content":"内容"}'}},))
            raise ProviderFailure('model_timeout', '模型暂时超时。')
        async def close(self):
            pass
    core = make_core(tmp_path, ToolProvider())
    result = run(core.chat('alice', ChatInput(text='写一条笔记')))
    assert result.status == 'model_timeout' and '已创建本地笔记：记录' in result.reply
    assert len(core.store.notes('alice')) == 1
    run(core.close())


def test_daily_followup_request_is_forced_through_tool_and_cannot_be_faked(tmp_path):
    class ToolProvider:
        forced = []
        async def generate_forced_tool(self, messages, tools, name):
            self.forced.append(name)
            return Generation('', 'test', tool_calls=({'id': 'daily', 'type': 'function',
                'function': {'name': 'set_daily_followup', 'arguments': '{"enabled":true,"hour":20,"minute":0}'}},))
        async def generate_with_tools(self, messages, tools):
            return Generation('每日主动关心已开启。', 'test')
        async def close(self):
            pass
    provider = ToolProvider()
    core = make_core(tmp_path, provider)
    result = run(core.chat('alice', ChatInput(text='请开启每天晚上20点的主动关心。')))
    assert result.status == 'completed' and provider.forced == ['set_daily_followup']
    assert core.store.followup('alice')['enabled'] == 1
    run(core.close())
    assert not check_output('每日主动关心已开启。')


def test_tool_round_limit_stops_repeated_mutation_and_keeps_receipts(tmp_path):
    class RepeatingProvider:
        calls = 0
        async def generate_with_tools(self, messages, tools):
            self.calls += 1
            return Generation('', 'test', tool_calls=({'id': f'call{self.calls}', 'type': 'function',
                'function': {'name': 'create_note', 'arguments': '{"title":"记录","content":"内容"}'}},))
        async def close(self):
            pass
    core = make_core(tmp_path, RepeatingProvider(), max_tool_rounds=2)
    result = run(core.chat('alice', ChatInput(text='写笔记')))
    assert result.status == 'tool_limit' and '已创建本地笔记' in result.reply
    assert len(core.store.notes('alice')) == 2
    run(core.close())


def test_tools_respect_no_save_and_user_isolation(tmp_path):
    core = make_core(tmp_path)
    result = run(core.execute_tool('alice', 'create_note', {'title': '标题', 'content': '私有内容'}))
    note_id = result['id']
    assert run(core.execute_tool('bob', 'get_note', {'id': note_id}))['ok'] is False
    core.set_mode('alice', True)
    assert run(core.execute_tool('alice', 'get_note', {'id': note_id}))['ok'] is False
    assert run(core.execute_tool('alice', 'create_note', {'title': '不保存标记', 'content': '私密'}))['ok'] is False
    assert run(core.execute_tool('alice', 'create_reminder', {'content': '不保存标记', 'due': '2099-01-01 12:00'}))['ok'] is False
    assert len(core.store.notes('alice')) == 1
    assert core.store.reminders('alice') == []
    run(core.close())
    assert b'\xe4\xb8\x8d\xe4\xbf\x9d\xe5\xad\x98\xe6\xa0\x87\xe8\xae\xb0' not in (tmp_path / 'chat.sqlite3').read_bytes()


def test_reminders_survive_restart_and_are_delivered_once_per_user(tmp_path):
    now = [1793102400.0]
    core = make_core(tmp_path, clock=lambda: now[0])
    due = core.local_time(now[0] + 60).isoformat()
    result = run(core.execute_tool('alice', 'create_reminder', {'content': '开会', 'due': due}))
    assert result['ok']
    run(core.close())
    now[0] += 65
    core = make_core(tmp_path, clock=lambda: now[0])
    core.tick()
    core.tick()
    assert core.inbox('bob') == []
    notifications = core.inbox('alice')
    assert len(notifications) == 1 and notifications[0]['content'] == '提醒：开会'
    core.acknowledge('alice', [notifications[0]['id']])
    run(core.close())
    core = make_core(tmp_path, clock=lambda: now[0])
    core.tick()
    assert core.inbox('alice') == []
    run(core.close())


def test_event_reminders_and_daily_followup_can_be_cancelled(tmp_path):
    now = [1793102400.0]
    core = make_core(tmp_path, clock=lambda: now[0])
    start = core.local_time(now[0] + 120).isoformat()
    event = run(core.execute_tool('alice', 'create_event', {'title': '面试', 'start': start, 'remind_before_minutes': 1}))
    assert event['ok'] and len(core.store.reminders('alice')) == 1
    assert run(core.execute_tool('bob', 'delete_event', {'id': event['id']}))['ok'] is False
    assert run(core.execute_tool('alice', 'delete_event', {'id': event['id']}))['ok']
    now[0] += 130
    core.tick()
    assert core.inbox('alice') == []
    local = core.local_time()
    core.set_followup('alice', FollowupConfig(enabled=True, hour=local.hour, minute=local.minute))
    core.tick()
    core.tick()
    assert len(core.inbox('alice')) == 1
    core.set_followup('alice', FollowupConfig(enabled=False))
    assert core.inbox('alice') == []
    due = core.local_time(now[0] + 30).isoformat()
    assert run(core.execute_tool('alice', 'schedule_followup', {'content': '面试怎么样？', 'due': due}))['ok']
    core.set_followup('alice', FollowupConfig(enabled=False))
    now[0] += 31
    core.tick()
    assert core.inbox('alice') == []
    run(core.close())


def test_daily_followup_is_not_sent_hours_late_or_while_user_is_active(tmp_path):
    zone = timezone(timedelta(hours=8))
    now = [datetime(2026, 10, 27, 22, 0, tzinfo=zone).timestamp()]
    core = make_core(tmp_path, clock=lambda: now[0])
    core.set_followup('alice', FollowupConfig(enabled=True, hour=20, minute=0))
    core.tick()
    assert core.inbox('alice') == []
    now[0] = datetime(2026, 10, 28, 19, 50, tzinfo=zone).timestamp()
    core.store.record_activity('alice', now[0])
    now[0] = datetime(2026, 10, 28, 20, 0, tzinfo=zone).timestamp()
    core.tick()
    assert core.inbox('alice') == []
    now[0] = datetime(2026, 10, 29, 20, 0, tzinfo=zone).timestamp()
    core.tick()
    assert len(core.inbox('alice')) == 1
    run(core.close())


def test_private_mode_pauses_delivery_without_losing_previous_reminder(tmp_path):
    now = [1793102400.0]
    core = make_core(tmp_path, clock=lambda: now[0])
    due = core.local_time(now[0] + 60).isoformat()
    assert run(core.execute_tool('alice', 'create_reminder', {'content': '合成提醒', 'due': due}))['ok']
    core.set_mode('alice', True)
    now[0] += 65
    core.tick()
    assert core.inbox('alice') == [] and core.store.reminders('alice')
    core.set_mode('alice', False)
    assert len(core.inbox('alice')) == 1
    run(core.close())
