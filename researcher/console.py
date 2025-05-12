from rich.console import Console as RichConsole
from rich.padding import Padding

console = RichConsole()


class Console:
    @staticmethod
    def welcome(args) -> None:
        console.rule("[bold blue]Researcher")
        console.print("[bold green]Welcome to researcher![/bold green]\n")
        
        # Format args in a more readable way
        console.print("[bold cyan]Configuration:[/bold cyan]")
        console.print(f"  [yellow]Model:[/yellow] {args.model_name}")
        console.print(f"  [yellow]Data path:[/yellow] {args.data_dir}")
        console.print(f"  [yellow]Max tokens:[/yellow] {args.max_tokens}")
        console.print()

    @staticmethod
    def step_start(name: str, color: str) -> None:
        console.rule(f"[bold {color}]Begin {name} step")

    @staticmethod
    def chat_response_start() -> None:
        pass

    @staticmethod
    def chat_message_content_delta(message_content_delta: str) -> None:
        console.print(message_content_delta, end="")

    @staticmethod
    def chat_response_complete() -> None:
        console.print("\n")

    @staticmethod
    def tool_call_start(tool_call: str) -> None:
        console.print(f"[bold yellow]Tool call: [/bold yellow]{tool_call}\n")

    @staticmethod
    def tool_call_complete(tool_response: str) -> None:
        lines = tool_response.split("\n")
        if len(lines) > 4:
            lines = lines[:4]
            lines.append("...")
            tool_response = "\n".join(lines)
        console.print(
            Padding.indent(f"{tool_response}\n", 4),
            no_wrap=True,
            overflow="ellipsis",
        )

    @staticmethod
    def user_input_complete(user_input: str) -> None:
        console.print()

    @staticmethod
    def print(message: str) -> None:
        console.print(message)