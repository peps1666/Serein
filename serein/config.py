from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import secrets

from dotenv import load_dotenv


@dataclass
class Settings:
    provider: str = "demo"
    users: dict[str, str] = field(default_factory=lambda: {"local": secrets.token_urlsafe(18)}, repr=False)
    db_path: Path = Path("data/serein.sqlite3")
    api_key: str = field(default="", repr=False)
    base_url: str = ""
    model: str = ""
    cloud_acknowledged: bool = False
    daily_call_limit: int = 100
    timeout_seconds: float = 20
    max_output_tokens: int = 400
    context_messages: int = 20
    context_characters: int = 12000
    idle_seconds: int = 1800
    session_seconds: int = 7200
    history_days: int = 7
    preference_days: int = 14
    search_api_key: str = field(default="", repr=False)
    daily_search_limit: int = 25
    timezone_offset_minutes: int = 480
    max_tool_rounds: int = 5

    def __post_init__(self):
        if not self.base_url:
            self.base_url = "https://api.deepseek.com" if self.provider == "deepseek" else "https://api.openai.com/v1"
        if self.provider not in {"demo", "openai", "deepseek"}:
            raise ValueError("SEREIN_PROVIDER must be demo, openai or deepseek")
        if not self.users or any(not u or not isinstance(p, str) or len(p) < 12 for u, p in self.users.items()):
            raise ValueError("Each login needs a username and a password of at least 12 characters")
        if self.provider != "demo" and not (
            self.api_key and self.model and self.cloud_acknowledged
        ):
            raise ValueError("Cloud mode requires a provider API key, SEREIN_MODEL and SEREIN_CLOUD_ACKNOWLEDGED=true")
        for name in ("daily_call_limit", "daily_search_limit", "timeout_seconds", "max_output_tokens", "idle_seconds", "session_seconds", "history_days", "preference_days"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.context_messages < 2 or self.context_characters < 2000:
            raise ValueError("Context requires at least 2 messages and 2000 characters")
        if not -720 <= self.timezone_offset_minutes <= 840 or not 1 <= self.max_tool_rounds <= 10:
            raise ValueError("Invalid timezone offset or tool round limit")

    @classmethod
    def from_env(cls, *, demo=False):
        # Resolve the project .env, independent of the caller's stack frame.
        load_dotenv(Path.cwd() / ".env", override=False)
        values = {}
        provider = "demo" if demo else os.getenv("SEREIN_PROVIDER", "auto").lower()
        if provider == "auto":
            if os.getenv("OPENAI_API_KEY") and os.getenv("SEREIN_MODEL"):
                provider = "openai"
            elif os.getenv("DEEPSEEK_API_KEY"):
                provider = "deepseek"
            else:
                raise ValueError("No model configured. Set DEEPSEEK_API_KEY or OPENAI_API_KEY + SEREIN_MODEL; use --demo explicitly for fixed replies.")
        values["provider"] = provider
        values["model"] = "" if provider == "demo" else os.getenv("SEREIN_MODEL", "deepseek-flash" if provider == "deepseek" else "")
        for name in ("daily_call_limit", "daily_search_limit", "max_output_tokens", "context_messages", "context_characters", "idle_seconds", "session_seconds", "history_days", "preference_days", "timezone_offset_minutes", "max_tool_rounds"):
            if value := os.getenv(f"SEREIN_{name.upper()}"):
                values[name] = int(value)
        if value := os.getenv("SEREIN_TIMEOUT_SECONDS"):
            values["timeout_seconds"] = float(value)
        if value := os.getenv("SEREIN_USERS"):
            users = json.loads(value)
            if not isinstance(users, dict):
                raise ValueError("SEREIN_USERS must be a JSON object")
            values["users"] = users
        values.update(
            db_path=Path(os.getenv("SEREIN_DB_PATH", "data/serein.sqlite3")),
            api_key=os.getenv("DEEPSEEK_API_KEY" if provider == "deepseek" else "OPENAI_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com") if provider == "deepseek" else os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            cloud_acknowledged=os.getenv("SEREIN_CLOUD_ACKNOWLEDGED", "").lower() == "true",
            search_api_key=os.getenv("BRAVE_SEARCH_API_KEY", ""),
        )
        return cls(**values)
