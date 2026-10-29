"""Direct local entry to the same companion core, independent of Gradio."""
import asyncio

from pydantic import ValidationError

from .config import Settings
from .core import AppError, Companion
from .models import ChatInput
from .provider import DemoProvider, OpenAIProvider
from .store import Store


async def chat_in_terminal(settings: Settings):
    core = Companion(settings, Store(settings.db_path),
                     DemoProvider() if settings.provider == "demo" else OpenAIProvider(settings))
    user = "local" if "local" in settings.users else next(iter(settings.users))
    print(f"Serein · {settings.provider} / {settings.model or 'offline demo'}")
    print("本机终端对话。/new 开始新对话，/private on 开启不保存，/exit 退出。")
    if settings.provider != "demo":
        print(f"对话将发送至配置的模型服务：{settings.base_url}")

    async def deliver_notifications():
        while True:
            for notification in core.inbox(user):
                print(f"\nSerein · {notification['content']}", flush=True)
                core.acknowledge(user, [notification['id']])
            await asyncio.sleep(5)

    notifier = asyncio.create_task(deliver_notifications())
    try:
        while True:
            try:
                text = (await asyncio.to_thread(input, "\n你：")).strip()
            except EOFError:
                break
            if text in {"/exit", "/quit"}:
                break
            if not text:
                continue
            if text == "/new":
                core.new_session(user)
                print("已开始新对话。")
                continue
            try:
                result = await core.chat(user, ChatInput(text=text))
                print("Serein：" + (result.reply or "这一轮已停止。"))
            except AppError as exc:
                print(exc.message)
            except ValidationError:
                print("请输入 1–2000 字的内容。")
    finally:
        notifier.cancel()
        await asyncio.gather(notifier, return_exceptions=True)
        await core.close()
