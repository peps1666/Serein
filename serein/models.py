from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ChatInput(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    request_id: UUID = Field(default_factory=uuid4)


class ChatResult(BaseModel):
    request_id: str
    status: str
    reply: str = ""
    elapsed_ms: int = 0
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class Preferences(BaseModel):
    nickname: str = Field(default="", max_length=20, pattern=r"^[\w\u4e00-\u9fff ·-]*$")
    reply_style: Literal["简短", "适中"] = "简短"
    support_style: Literal["先听我说", "可以一起想办法"] = "先听我说"


class PreferenceInput(Preferences):
    confirmed: Literal[True]


class ModeInput(BaseModel):
    no_save: bool


class DeleteInput(BaseModel):
    confirmed: Literal[True]


class ToolArgs(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')


class SearchArgs(ToolArgs):
    query: str = Field(min_length=1, max_length=128, description="要搜索的关键词或问题")


class NoteCreateArgs(ToolArgs):
    title: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1, max_length=4000)


class NoteQueryArgs(ToolArgs):
    query: str = Field(default="", max_length=100, description="按标题或内容筛选；空字符串列出最近笔记")


class IdArgs(ToolArgs):
    id: str = Field(min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$")


class EventCreateArgs(ToolArgs):
    title: str = Field(min_length=1, max_length=120)
    start: str = Field(min_length=10, max_length=40, description="ISO 8601 日期时间，缺省时区使用用户本地时区")
    description: str = Field(default="", max_length=2000)
    remind_before_minutes: int | None = Field(default=None, ge=0, le=10080, description="提前几分钟提醒；不提醒时为 null")


class EventQueryArgs(ToolArgs):
    start_date: str = Field(default="", max_length=10, description="查询开始日期 YYYY-MM-DD，默认今天")
    end_date: str = Field(default="", max_length=10, description="查询结束日期 YYYY-MM-DD，包含该日，默认 30 天后")


class ReminderCreateArgs(ToolArgs):
    content: str = Field(min_length=1, max_length=500)
    due: str = Field(min_length=10, max_length=40, description="ISO 8601 具体日期时间，缺省时区使用用户本地时区")


class FollowupConfig(ToolArgs):
    enabled: bool
    hour: int = Field(default=20, ge=0, le=23)
    minute: int = Field(default=0, ge=0, le=59)


class NotificationAck(BaseModel):
    ids: list[str] = Field(max_length=100)


class ToolInput(BaseModel):
    arguments: dict = Field(default_factory=dict)
