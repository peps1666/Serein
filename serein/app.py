import asyncio
from contextlib import asynccontextmanager, suppress
import logging
import os
from pathlib import Path
import secrets

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from .config import Settings
from .core import AppError, Companion
from .models import (ChatInput, DeleteInput, ModeInput, PreferenceInput, Preferences,
                     FollowupConfig, NotificationAck, ToolInput)
from .provider import DemoProvider, OpenAIProvider
from .store import Store


def create_app(settings=None, provider=None, with_ui=True):
    settings = settings or Settings.from_env()
    core = Companion(settings, Store(settings.db_path), provider or (
        DemoProvider() if settings.provider == "demo" else OpenAIProvider(settings)))

    @asynccontextmanager
    async def lifespan(app):
        async def cleanup():
            ticks = 0
            while True:
                try:
                    core.tick()
                    if ticks % 6 == 0:
                        core.maintenance()
                except Exception:
                    # Never log exception details: they can include user content.
                    logging.getLogger(__name__).warning('后台提醒处理暂时失败，将在下一轮重试。')
                ticks += 1
                await asyncio.sleep(5)
        cleaner = asyncio.create_task(cleanup())
        try:
            yield
        finally:
            cleaner.cancel()
            with suppress(asyncio.CancelledError):
                await cleaner
            await core.close()

    app = FastAPI(title="Serein 本地原型", lifespan=lifespan)
    app.state.core = core
    security = HTTPBasic(auto_error=False)

    def valid_login(user, password):
        expected = settings.users.get(user)
        return expected is not None and secrets.compare_digest(expected.encode(), password.encode())

    async def current_user(credentials: HTTPBasicCredentials | None = Depends(security)):
        if credentials is None or not valid_login(credentials.username, credentials.password):
            raise AppError(401, "请使用已授权的用户名与密码。")
        return credentials.username

    @app.exception_handler(AppError)
    async def handle_error(request, exc):
        return JSONResponse({"detail": exc.message}, status_code=exc.status,
                            headers={"WWW-Authenticate": "Basic"} if exc.status == 401 else {})

    @app.middleware("http")
    async def no_cache(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/")
    async def root():
        return RedirectResponse("/chat/")

    @app.get("/health")
    async def health():
        return {"status": "ok", "provider": settings.provider, "model": settings.model,
                "real_model": settings.provider != "demo", "prototype": True}

    @app.get("/api/state")
    async def state(user=Depends(current_user)):
        return {"no_save": core.state(user).no_save, "history": core.history(user),
                "preferences": core.preferences(user), "provider": settings.provider}

    @app.post("/api/chat")
    async def chat(body: ChatInput, user=Depends(current_user)):
        return await core.chat(user, body)

    @app.post("/api/stop")
    async def stop(user=Depends(current_user)):
        core.stop(user)
        return {"status": "stopped"}

    @app.post("/api/session")
    async def session(user=Depends(current_user)):
        core.new_session(user)
        return {"status": "new_session"}

    @app.put("/api/mode")
    async def mode(body: ModeInput, user=Depends(current_user)):
        core.set_mode(user, body.no_save)
        return {"no_save": body.no_save}

    @app.put("/api/preferences")
    async def preferences(body: PreferenceInput, user=Depends(current_user)):
        return core.save_preferences(user, Preferences(**body.model_dump()))

    @app.delete("/api/preferences")
    async def forget(user=Depends(current_user)):
        core.forget_preferences(user)
        return {"status": "deleted"}

    @app.post("/api/delete-all")
    async def delete(body: DeleteInput, user=Depends(current_user)):
        core.delete_all(user)
        return {"status": "deleted"}

    @app.post("/api/tools/{name}")
    async def tool(name: str, body: ToolInput, user=Depends(current_user)):
        return await core.execute_tool(user, name, body.arguments)

    @app.get("/api/notifications")
    async def notifications(user=Depends(current_user)):
        return {"notifications": core.inbox(user)}

    @app.post("/api/notifications/ack")
    async def acknowledge(body: NotificationAck, user=Depends(current_user)):
        core.acknowledge(user, body.ids)
        return {"status": "acknowledged"}

    @app.get("/api/followup")
    async def get_followup(user=Depends(current_user)):
        return core.followup_settings(user)

    @app.put("/api/followup")
    async def set_followup(body: FollowupConfig, user=Depends(current_user)):
        core.invalidate(user)
        return core.set_followup(user, body)

    if with_ui:
        # Disable telemetry before importing Gradio. No browser history, examples
        # cache, flagging, file inputs, or durable Gradio State is used.
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
        os.environ["GRADIO_TEMP_DIR"] = str(settings.db_path.parent.resolve() / "gradio-tmp")
        import gradio as gr
        from .ui import build_ui
        demo = build_ui(core)
        app = gr.mount_gradio_app(app, demo, path="/chat", auth=valid_login,
                                  auth_message="Serein · 本地实验原型。请使用启动终端显示的账号登录。",
                                  blocked_paths=[str(settings.db_path.parent.resolve()), str(Path('.env').resolve())],
                                  show_error=False, enable_monitoring=False)
    return app
