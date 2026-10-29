import json

import gradio as gr
from pydantic import ValidationError

from .core import AppError, Companion
from .models import ChatInput, Preferences, FollowupConfig


def build_ui(core: Companion):
    demo_mode = core.settings.provider == "demo"

    def user(request):
        core.authorize(request.username)
        return request.username

    def snapshot(request: gr.Request):
        name = user(request)
        state = core.state(name)
        preferences = core.preferences(name)
        label = "不保存：仅内存上下文，当前不读取偏好" if state.no_save else "普通模式：对话最多保存 7 天"
        if not state.no_save:
            label = f"普通模式：对话最多保存 {core.settings.history_days} 天"
        return core.history(name), state.no_save, label, json.dumps(preferences or {}, ensure_ascii=False, indent=2)

    async def send(text, request: gr.Request):
        try:
            result = await core.chat(user(request), ChatInput(text=text))
            status = {"completed": "已回应", "cancelled": "已停止", "control": "控制已执行"}.get(result.status, "本次未完成")
            history, mode, label, preferences = snapshot(request)
            if result.status != "completed" and result.reply:
                status += " · " + result.reply
            remaining_text = "" if result.status in {"completed", "control", "cancelled"} else text
            return remaining_text, history, mode, label + " · " + status, preferences
        except (AppError, ValidationError) as exc:
            message = exc.message if isinstance(exc, AppError) else "请输入 1–2000 字的内容。"
            raise gr.Error(message) from None

    def switch_mode(no_save, request: gr.Request):
        core.set_mode(user(request), no_save)
        return snapshot(request)

    def stop(request: gr.Request):
        core.stop(user(request))
        return "已停止当前生成；已交给界面的回复无法撤回。"

    def new_session(request: gr.Request):
        core.new_session(user(request))
        return snapshot(request)

    def save(nickname, reply_style, support_style, confirmed, request: gr.Request):
        if not confirmed:
            raise gr.Error("请确认要保存上述三项偏好。")
        try:
            core.save_preferences(user(request), Preferences(nickname=nickname, reply_style=reply_style, support_style=support_style))
        except (AppError, ValidationError) as exc:
            raise gr.Error(exc.message if isinstance(exc, AppError) else "称呼限 20 字，请使用文字、数字或空格。") from None
        history, mode, label, preferences = snapshot(request)
        return history, mode, label + " · 偏好已保存", preferences, False

    def forget(request: gr.Request):
        core.forget_preferences(user(request))
        return snapshot(request)

    def delete(confirmed, request: gr.Request):
        if not confirmed:
            raise gr.Error("请勾选确认后再删除。")
        core.delete_all(user(request))
        history, mode, label, preferences = snapshot(request)
        return history, mode, label + " · 本方数据已删除", preferences, False, "", False, False, 20, 0

    async def refresh_tools(request: gr.Request):
        name = user(request)
        inbox = core.inbox(name)
        if core.state(name).no_save:
            return "不保存模式：暂不读取本地笔记、日程和提醒。", "不保存模式：提醒暂不显示。"
        notes = await core.execute_tool(name, 'find_notes', {})
        events = await core.execute_tool(name, 'list_events', {})
        reminders = await core.execute_tool(name, 'list_reminders', {})
        details = {'notes': notes.get('notes', []), 'next_30_days': events.get('events', []),
                   'scheduled_reminders': reminders.get('reminders', [])}
        return json.dumps(details, ensure_ascii=False, indent=2), json.dumps(inbox, ensure_ascii=False, indent=2)

    def followup_snapshot(request: gr.Request):
        settings = core.followup_settings(user(request))
        return settings['enabled'], settings['hour'], settings['minute']

    async def execute(name, arguments, request):
        result = await core.execute_tool(user(request), name, arguments)
        details, inbox = await refresh_tools(request)
        return result.get('receipt') or ("操作已完成。" if result.get('ok') else result.get('error', '操作未完成。')), details, inbox

    async def create_note(title, content, request: gr.Request):
        return await execute('create_note', {'title': title, 'content': content}, request)

    async def create_event(title, start, description, remind, request: gr.Request):
        arguments = {'title': title, 'start': start, 'description': description}
        if remind:
            arguments['remind_before_minutes'] = 0
        return await execute('create_event', arguments, request)

    async def create_reminder(content, due, request: gr.Request):
        return await execute('create_reminder', {'content': content, 'due': due}, request)

    async def search(query, request: gr.Request):
        result = await core.execute_tool(user(request), 'web_search', {'query': query})
        return json.dumps(result, ensure_ascii=False, indent=2)

    async def configure_followup(enabled, hour, minute, request: gr.Request):
        try:
            settings = FollowupConfig(enabled=enabled, hour=hour, minute=minute)
            core.invalidate(user(request))
            core.set_followup(user(request), settings)
            details, inbox = await refresh_tools(request)
            return ('已开启每日主动关心。' if enabled else '已关闭主动关心并取消待发跟进。'), details, inbox
        except (AppError, ValidationError) as exc:
            raise gr.Error(exc.message if isinstance(exc, AppError) else '请填写有效的小时和分钟。') from None

    async def acknowledge(request: gr.Request):
        name = user(request)
        notifications = core.inbox(name)
        core.acknowledge(name, [item['id'] for item in notifications])
        return await refresh_tools(request)

    with gr.Blocks(title="Serein · 听你说", analytics_enabled=False, theme=gr.themes.Soft(), delete_cache=(60, 60)) as demo:
        gr.Markdown("# Serein\n给今天的心事，留一点空间。")
        gr.Markdown("**本地演示模式 · 固定示例回复，不调用模型**" if demo_mode else
                    f"**真实模型对话 · {core.settings.provider} / {core.settings.model}**\n对话会发送至 {core.settings.base_url}；不保存模式仅控制本方存储。")
        gr.Markdown("这是日常情绪支持 AI 原型。你可以倾诉、安静待一会儿，或在愿意时一起想办法。")
        with gr.Row():
            with gr.Column(scale=3):
                status = gr.Markdown()
                chatbot = gr.Chatbot(type="messages", height=410, label="听你说", show_copy_button=False,
                                     placeholder="今天怎么样？想说多少都可以。")
                text = gr.Textbox(label="你的话", placeholder="例如：今天被领导批评了，我只想吐槽一下。", lines=2, max_lines=5)
                with gr.Row():
                    submit = gr.Button("发送", variant="primary")
                    stop_button = gr.Button("停止回复")
                    fresh = gr.Button("开始新对话")
                gr.Examples([["今天被领导批评了，我觉得自己特别没用。"], ["别分析，我只想吐槽。"], ["今天想安静待一会儿，少说一点。"]],
                            inputs=text, cache_examples=False, label="用合成场景试一试")
            with gr.Column(scale=2):
                no_save = gr.Checkbox(label="开启不保存模式", value=False)
                gr.Markdown(f"不保存时，原文只在运行中的内存里；空闲 {core.settings.idle_seconds // 60} 分钟或开始新对话后清空，最长 {core.settings.session_seconds // 60} 分钟。切换模式会清空当前上下文。已有历史需单独删除。云模型与浏览器的数据边界见说明。")
                with gr.Accordion("我希望你这样陪我", open=True):
                    nickname = gr.Textbox(label="怎么称呼你（可留空）", max_lines=1)
                    reply_style = gr.Radio(["简短", "适中"], value="简短", label="回复长度")
                    support_style = gr.Radio(["先听我说", "可以一起想办法"], value="先听我说", label="倾听方式")
                    confirm = gr.Checkbox(label=f"确认保存以上三项，最多 {core.settings.preference_days} 天，可随时撤回")
                    save_button = gr.Button("确认保存偏好")
                    memory = gr.Code(label="已确认的偏好与凭据", language="json", interactive=False)
                    forget_button = gr.Button("撤回全部偏好")
                with gr.Accordion("数据控制与说明", open=False):
                    gr.Markdown("开始新对话仅清空上下文。删除全部数据会清理本方对话、偏好、笔记、日程、提醒和跟进设置；无法删除模型服务或浏览器已持有的副本。同一账号的标签页共享控制状态。")
                    delete_confirm = gr.Checkbox(label="确认删除本方全部数据")
                    delete_button = gr.Button("删除全部数据", variant="stop")
                    gr.Markdown("固定命令：`查看记忆`、`停止当前回复`、`开启不保存`、`关闭不保存`。其他表达按普通聊天处理。")
        with gr.Accordion("提醒与主动跟进", open=True):
            gr.Markdown("提醒在服务运行时进入本地收件箱，页面打开时自动刷新；服务停止期间错过的提醒会在重启后显示。")
            inbox = gr.Code(label="未读提醒和跟进", language="json", interactive=False)
            with gr.Row():
                ack_button = gr.Button("标记全部已读")
                browser_notifications = gr.Button("开启浏览器通知")
            gr.Markdown("可选开启浏览器通知；通知只显示有新提醒，不展示内容。需要浏览器保持打开并允许通知。")
            with gr.Row():
                follow_enabled = gr.Checkbox(label="开启每日主动关心", value=False)
                follow_hour = gr.Number(label="小时", value=20, minimum=0, maximum=23, precision=0)
                follow_minute = gr.Number(label="分钟", value=0, minimum=0, maximum=59, precision=0)
                follow_save = gr.Button("保存跟进设置")
            follow_status = gr.Markdown()
            gr.Markdown("默认关闭；每天最多一条。如果指定时间内服务未运行或刚聊过天，当天不补发。关闭会取消待发的主动跟进。")
        with gr.Accordion("搜索、笔记与本地日历", open=False):
            gr.Markdown("也可以直接在对话中要求创建、查询或取消。网页搜索会将查询发送给所配置的搜索服务：Brave 或 DeepSeek。时间按配置的本地时区填写，例如 `2026-11-01 09:00`。")
            search_query = gr.Textbox(label="搜索网页")
            search_button = gr.Button("搜索")
            search_result = gr.Code(label="搜索结果", language="json", interactive=False)
            with gr.Tab("笔记"):
                note_title = gr.Textbox(label="笔记标题")
                note_content = gr.Textbox(label="笔记内容", lines=3)
                note_button = gr.Button("保存笔记")
            with gr.Tab("日程"):
                event_title = gr.Textbox(label="日程标题")
                event_time = gr.Textbox(label="开始时间")
                event_description = gr.Textbox(label="备注（可留空）")
                event_reminder = gr.Checkbox(label="开始时提醒")
                event_button = gr.Button("创建日程")
            with gr.Tab("提醒"):
                reminder_content = gr.Textbox(label="提醒内容")
                reminder_time = gr.Textbox(label="提醒时间")
                reminder_button = gr.Button("设置提醒")
            tool_status = gr.Markdown()
            tool_data = gr.Code(label="笔记、未来日程及待发提醒", language="json", interactive=False)
        outputs = [chatbot, no_save, status, memory]
        demo.load(snapshot, outputs=outputs, api_name=False)
        submit.click(send, inputs=text, outputs=[text, *outputs], concurrency_limit=5, concurrency_id="chat", api_name="send")
        text.submit(send, inputs=text, outputs=[text, *outputs], concurrency_limit=5, concurrency_id="chat", api_name=False)
        # These controls bypass the generation queue. The core cancels generation
        # and checks its epoch before committing; Gradio cancellation alone isn't relied on.
        stop_button.click(stop, outputs=status, queue=False, api_name=False)
        fresh.click(new_session, outputs=outputs, queue=False, api_name=False)
        no_save.input(switch_mode, inputs=no_save, outputs=outputs, queue=False, api_name=False)
        save_button.click(save, inputs=[nickname, reply_style, support_style, confirm], outputs=[*outputs, confirm], queue=False, api_name=False)
        forget_button.click(forget, outputs=outputs, queue=False, api_name=False)
        delete_button.click(delete, inputs=delete_confirm,
                            outputs=[*outputs, delete_confirm, text, confirm, follow_enabled, follow_hour, follow_minute],
                            queue=False, api_name=False)
        demo.load(refresh_tools, outputs=[tool_data, inbox], api_name=False)
        demo.load(followup_snapshot, outputs=[follow_enabled, follow_hour, follow_minute], api_name=False)
        poll = gr.Timer(5).tick(refresh_tools, outputs=[tool_data, inbox], queue=False, api_name=False)
        poll.then(fn=None, inputs=inbox, outputs=None, queue=False, api_name=False, js="""(payload) => {
            if (!('Notification' in window) || Notification.permission !== 'granted') return;
            try {
                const items = JSON.parse(payload);
                if (!Array.isArray(items)) return;
                window.sereinNoticeIds = window.sereinNoticeIds || new Set();
                const fresh = items.filter(item => item && item.id && !window.sereinNoticeIds.has(item.id));
                fresh.forEach(item => window.sereinNoticeIds.add(item.id));
                if (fresh.length) new Notification('Serein', {body: '你有新的提醒，请打开 Serein 查看。'});
            } catch (_) { /* no notifications while private mode is active */ }
        }""")
        browser_notifications.click(fn=None, inputs=None, outputs=None, queue=False, api_name=False,
                                    js="() => { if ('Notification' in window) Notification.requestPermission(); }")
        ack_button.click(acknowledge, outputs=[tool_data, inbox], queue=False, api_name=False)
        follow_save.click(configure_followup, inputs=[follow_enabled, follow_hour, follow_minute],
                          outputs=[follow_status, tool_data, inbox], queue=False, api_name=False)
        note_button.click(create_note, inputs=[note_title, note_content], outputs=[tool_status, tool_data, inbox], api_name=False)
        event_button.click(create_event, inputs=[event_title, event_time, event_description, event_reminder],
                           outputs=[tool_status, tool_data, inbox], api_name=False)
        reminder_button.click(create_reminder, inputs=[reminder_content, reminder_time],
                              outputs=[tool_status, tool_data, inbox], api_name=False)
        search_button.click(search, inputs=search_query, outputs=search_result, api_name=False)
    return demo.queue(max_size=20, default_concurrency_limit=5)
