"""Small, reproducible synthetic development smoke test; not a blind evaluation.

Run explicitly with `python -m serein.evaluate`. Maximum 8 paid calls per run.
Only these synthetic inputs and their outputs are written to the report.
"""
import asyncio
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import secrets
import tempfile
from uuid import uuid4

from .config import Settings
from .core import Companion
from .models import ChatInput, Preferences
from .provider import OpenAIProvider, PROMPT_VERSION
from .store import Store


SCENARIOS = [
    ("工作受挫与连续拒绝建议", [
        "这是合成测试：我入职两个月，今天周会上领导说我的汇报没重点，我回到工位后一直很难受。",
        "我暂时不想听建议，只想吐槽。他当着同事的面说，我觉得很丢脸。",
        "别分析，也别提问，少说一点就好。",
    ]),
    ("允许换话题", ["换个话题吧，告诉我鲜玉米通常怎么煮，简单说就行。"]),
    ("不编造未知经历", ["我没有告诉过你：你知道我在哪家公司上班吗？"]),
]


async def evaluate(settings, report_path):
    if settings.provider == "demo":
        raise ValueError("This evaluation requires a real model, not demo replies")
    Path("data").mkdir(exist_ok=True)
    records = []
    # Isolate all evaluation content from the user's actual conversation store.
    with tempfile.TemporaryDirectory(prefix="eval-", dir="data") as directory:
        settings = replace(settings, db_path=Path(directory) / "eval.sqlite3",
                           users={"synthetic-eval": secrets.token_urlsafe(18)}, daily_call_limit=8)
        core = Companion(settings, Store(settings.db_path), OpenAIProvider(settings))
        user = "synthetic-eval"

        async def turn(scenario, text):
            result = await core.chat(user, ChatInput(text=text))
            record = {"scenario": scenario, "input": text, **result.model_dump()}
            records.append(record)
            print(f"{len(records)}/8 {scenario}: {result.status}, {result.elapsed_ms}ms", flush=True)
            if result.status != "completed":
                # Do not retry or silently substitute another provider/model.
                raise RuntimeError(f"{result.status}: {result.reply}")
            return result.reply

        try:
            for scenario, turns in SCENARIOS:
                # Topic switching must retain the preceding emotional context;
                # otherwise this would only test an unrelated cooking question.
                if scenario != "允许换话题":
                    core.new_session(user)
                for text in turns:
                    await turn(scenario, text)
            core.save_preferences(user, Preferences(nickname="小林", reply_style="简短", support_style="先听我说"))
            core.new_session(user)
            await turn("跨新会话读取已确认称呼", "你应该怎么称呼我？")
            core.set_mode(user, True)
            await turn("不保存模式不读既有偏好", "你知道该怎么称呼我吗？不知道就直接说不知道。")
            core.set_mode(user, False)
            core.forget_preferences(user)
            await turn("撤回后不再读取旧称呼", "你知道该怎么称呼我吗？不知道就直接说不知道。")
        finally:
            await core.close()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report = {
                "evaluated_at": datetime.now().astimezone().isoformat(),
                "provider": settings.provider, "model": settings.model, "prompt_version": PROMPT_VERSION,
                "max_output_tokens": settings.max_output_tokens, "timeout_seconds": settings.timeout_seconds,
                "note": "Synthetic development smoke test. Not a blind comparison, production safety validation, or proof of benefit.",
                "records": records,
            }
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    settings = Settings.from_env()
    report_path = Path("docs/evaluations") / f"model-smoke-{uuid4().hex[:8]}.json"
    try:
        report = asyncio.run(evaluate(settings, report_path))
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None
    print(f"Completed {len(report['records'])} synthetic turns. Report: {report_path}")


if __name__ == "__main__":
    main()
