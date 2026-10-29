import asyncio
from dataclasses import dataclass
import logging
from typing import Protocol

from .config import Settings
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT


@dataclass(frozen=True)
class Generation:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: tuple[dict, ...] = ()


class ProviderFailure(Exception):
    """Only stable codes and reviewed messages cross the provider boundary."""

    def __init__(self, code, message):
        self.code, self.message = code, message
        super().__init__(code)


class Provider(Protocol):
    async def generate(self, messages: list[dict]) -> str | Generation: ...
    async def close(self) -> None: ...


class DemoProvider:
    async def generate(self, messages):
        await asyncio.sleep(0.3)
        text = messages[-1]["content"]
        if any(word in text for word in ("别分析", "不想听建议", "只想吐槽", "不要建议")):
            return "好，先不分析，也不提建议。你想说多少都可以，我听着。"
        if any(word in text for word in ("少说", "安静", "陪着")):
            return "好，我在这里。"
        if any(word in text for word in ("怎么办", "想办法")):
            return "如果你愿意，我们先挑眼前最困扰你的一件事。你希望先从哪里开始？"
        if any(word in text for word in ("领导", "工作", "加班", "没用")):
            return "今天在工作里受了委屈，听起来挺不好受的。一次不顺利，不代表你这个人没有价值。"
        return "我听着。你可以按自己的节奏说，不用急着把感受整理清楚。"

    async def close(self):
        pass


class OpenAIProvider:
    def __init__(self, settings: Settings):
        from openai import AsyncOpenAI

        # Avoid SDK and HTTP debug logs containing bodies; do not inherit OPENAI_LOG.
        for name in ("openai", "httpx", "httpcore"):
            logging.getLogger(name).setLevel(logging.WARNING)
        self.client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url,
                                  timeout=settings.timeout_seconds, max_retries=0)
        self.model = settings.model
        self.provider = settings.provider
        self.max_tokens = settings.max_output_tokens

    async def generate(self, messages):
        return await self._generate(messages)

    async def generate_with_tools(self, messages, tools):
        return await self._generate(messages, tools)

    async def generate_forced_tool(self, messages, tools, name):
        return await self._generate(messages, tools, name)

    async def _generate(self, messages, tools=None, force_tool=None):
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        # DeepSeek's documented schema has no store parameter. Disable its
        # default thinking mode so short support replies don't spend the token
        # limit reasoning and return an empty answer.
        options = {"extra_body": {"thinking": {"type": "disabled"}}} if self.provider == "deepseek" else {"store": False}
        try:
            if tools is not None:
                options["tools"] = tools
            if force_tool:
                options["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
            result = await self.client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=self.max_tokens, **options,
            )
        except APITimeoutError:
            raise ProviderFailure("model_timeout", "模型回复超时了，这一轮没有完成。你可以稍后重试。") from None
        except APIConnectionError:
            raise ProviderFailure("model_connection_error", "暂时连不上模型服务，请检查网络或稍后重试。") from None
        except APIStatusError as exc:
            errors = {
                400: ("model_config_error", "模型参数不兼容，请检查服务地址、模型名称和参数。"),
                401: ("model_auth_error", "模型密钥未通过验证，请检查本机模型配置。"),
                402: ("model_balance_error", "模型服务余额不足，请补充额度后再试。"),
                403: ("model_auth_error", "当前密钥没有模型访问权限，请检查本机模型配置。"),
                404: ("model_config_error", "未找到配置的模型或接口，请检查模型名称和服务地址。"),
                429: ("model_rate_limited", "模型服务暂时限流，请稍后再试。"),
            }
            code, message = errors.get(exc.status_code, ("model_unavailable", "模型服务暂时不可用，你可以稍后重试。"))
            raise ProviderFailure(code, message) from None
        if not result.choices or result.choices[0].finish_reason not in {"stop", "tool_calls"}:
            raise ProviderFailure("model_incomplete", "这一轮回复没有完整生成，暂时不能显示。你可以稍后重试。")
        usage = result.usage
        message = result.choices[0].message
        calls = tuple(call.model_dump(exclude_none=True) for call in (message.tool_calls or []))
        if result.choices[0].finish_reason == "tool_calls" and not calls:
            raise ProviderFailure("model_incomplete", "模型没有提供完整的工具调用，请稍后重试。")
        return Generation(text=message.content or "", model=result.model,
                          input_tokens=usage.prompt_tokens if usage else 0,
                          output_tokens=usage.completion_tokens if usage else 0, tool_calls=calls)

    async def close(self):
        await self.client.close()


FALLBACK = "这次暂时没能好好回应你。你可以稍后再试；查看偏好、停止和删除仍然可用。"


def check_output(text, actions=()):
    """Minimal local checks, not a validated comprehensive safety classifier."""
    forbidden = ("只有我懂你", "你只需要我", "不要联系任何人", "你患有", "建议你服用",
                 "我已经记住", "已记住", "已经联系")
    if not isinstance(text, str) or not text.strip() or len(text) > 4000:
        return False
    if any(term in text for term in forbidden):
        return False
    if any(term in text for term in ("已经保存", "已保存", "已经创建", "已创建", "已经设置", "已设置", "已经添加", "已添加", "已开启", "已经开启", "开启了", "设置好了", "已安排", "已经安排")) and "save" not in actions:
        return False
    if any(term in text for term in ("已经删除", "已删除", "已经取消", "已取消", "已关闭", "已经关闭", "关闭了")) and "delete" not in actions:
        return False
    return True
