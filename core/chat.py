from core.claude import Claude
from mcp_client import MCPClient
from core.tools import ToolManager
from anthropic.types import MessageParam


class Chat:
    # Core chat orchestrator:
    # - stores conversation history
    # - sends messages + tool schemas to Claude
    # - executes tool calls when requested
    # - loops until model returns a final text answer
    def __init__(self, claude_service: Claude, clients: dict[str, MCPClient]):
        # Wrapper around Anthropic API operations.
        self.claude_service: Claude = claude_service
        # MCP clients keyed by id; tools are discovered from these servers.
        self.clients: dict[str, MCPClient] = clients
        # Shared conversation history sent to the model each turn.
        self.messages: list[MessageParam] = []

    async def _process_query(self, query: str):
        # Base behavior: append raw user query.
        # Subclasses (like CliChat) can override to add preprocessing/context.
        self.messages.append({"role": "user", "content": query})

    async def run(
        self,
        query: str,
    ) -> str:
        # Collects the final plain-text answer that we return to caller.
        final_text_response = ""

        # Convert user input into one or more messages for this turn.
        await self._process_query(query)

        # Tool loop:
        # 1) ask model
        # 2) if model requests tools, execute them and feed results back
        # 3) repeat until model returns a non-tool final answer
        while True:
            response = self.claude_service.chat(
                messages=self.messages,
                tools=await ToolManager.get_all_tools(self.clients),
            )

            # Persist assistant message (including tool-use blocks when present).
            self.claude_service.add_assistant_message(self.messages, response)

            if response.stop_reason == "tool_use":
                # Prints any assistant explanatory text before tool execution.
                print(self.claude_service.text_from_message(response))
                tool_result_parts = await ToolManager.execute_tool_requests(
                    self.clients, response
                )

                # Tool results are added as a user-role tool_result message,
                # enabling the model to continue reasoning with fresh outputs.
                self.claude_service.add_user_message(
                    self.messages, tool_result_parts
                )
            else:
                # Model finished without requesting additional tools.
                final_text_response = self.claude_service.text_from_message(
                    response
                )
                break

        return final_text_response
