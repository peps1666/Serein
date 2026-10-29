import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Run the Serein local prototype")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--demo", action="store_true", help="Explicitly use offline fixed replies")
    parser.add_argument("--terminal", action="store_true", help="Chat in this terminal without starting a web server")
    args = parser.parse_args()
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    from .config import Settings
    try:
        settings = Settings.from_env(demo=args.demo)
    except ValueError as exc:
        parser.error(str(exc))
    if args.terminal:
        import asyncio
        from .terminal import chat_in_terminal
        try:
            asyncio.run(chat_in_terminal(settings))
        except KeyboardInterrupt:
            print("\n已结束对话。")
        return
    from .app import create_app
    import uvicorn
    app = create_app(settings)
    print(f"\nSerein ({settings.provider}, {settings.model or 'offline demo'})  http://127.0.0.1:{args.port}/chat/")
    if not os.getenv("SEREIN_USERS"):
        print(f"Local login: local\nPassword: {settings.users['local']}\n")
    else:
        print("Login with an account configured in SEREIN_USERS.\n")
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
