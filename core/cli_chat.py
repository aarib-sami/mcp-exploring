from typing import List, Tuple
from mcp.types import Prompt, PromptMessage
from anthropic.types import MessageParam

from core.chat import Chat
from core.claude import Claude
from mcp_client import MCPClient


class CliChat(Chat):
    # CliChat extends the base Chat flow with CLI-specific behavior:
    # - slash commands ("/prompt_name doc_id")
    # - @document mentions ("@report.pdf")
    # - document resource fetching from the MCP doc server
    def __init__(
        self,
        doc_client: MCPClient,
        clients: dict[str, MCPClient],
        claude_service: Claude,
    ):
        # Base Chat manages shared message history, tool routing, and model calls.
        super().__init__(clients=clients, claude_service=claude_service)

        # Dedicated client for doc resources/prompts (the default document server).
        self.doc_client: MCPClient = doc_client

    async def list_prompts(self) -> list[Prompt]:
        # Prompt templates registered by the document MCP server.
        return await self.doc_client.list_prompts()

    async def list_docs_ids(self) -> list[str]:
        # Reads document id list from resource URI declared in mcp_server.py.
        return await self.doc_client.read_resource("docs://documents")

    async def get_doc_content(self, doc_id: str) -> str:
        # Reads one document's contents from resource URI.
        return await self.doc_client.read_resource(f"docs://documents/{doc_id}")

    async def get_prompt(
        self, command: str, doc_id: str
    ) -> list[PromptMessage]:
        # Resolves one server-side prompt template with prompt args.
        return await self.doc_client.get_prompt(command, {"doc_id": doc_id})

    async def _extract_resources(self, query: str) -> str:
        # Mentions are words prefixed with '@', e.g. "@report.pdf".
        # We strip '@' and compare against known document ids.
        mentions = [word[1:] for word in query.split() if word.startswith("@")]

        doc_ids = await self.list_docs_ids()
        mentioned_docs: list[Tuple[str, str]] = []

        for doc_id in doc_ids:
            if doc_id in mentions:
                content = await self.get_doc_content(doc_id)
                mentioned_docs.append((doc_id, content))

        # Convert docs into tagged context blocks injected into the user prompt.
        return "".join(
            f'\n<document id="{doc_id}">\n{content}\n</document>\n'
            for doc_id, content in mentioned_docs
        )

    async def _process_command(self, query: str) -> bool:
        # Slash commands map to MCP prompts, e.g. "/format plan.md".
        if not query.startswith("/"):
            return False

        words = query.split()
        command = words[0].replace("/", "")

        messages = await self.doc_client.get_prompt(
            command, {"doc_id": words[1]}
        )

        # Convert MCP PromptMessage format into Anthropic MessageParam format.
        self.messages += convert_prompt_messages_to_message_params(messages)
        return True

    async def _process_query(self, query: str):
        # If input is a slash command, it becomes prompt messages directly.
        if await self._process_command(query):
            return

        # Otherwise, treat as normal user text and enrich with @mentioned docs.
        added_resources = await self._extract_resources(query)

        prompt = f"""
        The user has a question:
        <query>
        {query}
        </query>

        The following context may be useful in answering their question:
        <context>
        {added_resources}
        </context>

        Note the user's query might contain references to documents like "@report.docx". The "@" is only
        included as a way of mentioning the doc. The actual name of the document would be "report.docx".
        If the document content is included in this prompt, you don't need to use an additional tool to read the document.
        Answer the user's question directly and concisely. Start with the exact information they need. 
        Don't refer to or mention the provided context in any way - just use it to inform your answer.
        """

        # Base Chat later sends this message to the model.
        self.messages.append({"role": "user", "content": prompt})


def convert_prompt_message_to_message_param(
    prompt_message: "PromptMessage",
) -> MessageParam:
    # Normalize MCP prompt role names to Anthropic message roles.
    role = "user" if prompt_message.role == "user" else "assistant"

    content = prompt_message.content

    # Check if content is a dict-like object with a "type" field
    if isinstance(content, dict) or hasattr(content, "__dict__"):
        content_type = (
            content.get("type", None)
            if isinstance(content, dict)
            else getattr(content, "type", None)
        )
        if content_type == "text":
            content_text = (
                content.get("text", "")
                if isinstance(content, dict)
                else getattr(content, "text", "")
            )
            return {"role": role, "content": content_text}

    if isinstance(content, list):
        text_blocks = []
        for item in content:
            # Check if item is a dict-like object with a "type" field
            if isinstance(item, dict) or hasattr(item, "__dict__"):
                item_type = (
                    item.get("type", None)
                    if isinstance(item, dict)
                    else getattr(item, "type", None)
                )
                if item_type == "text":
                    item_text = (
                        item.get("text", "")
                        if isinstance(item, dict)
                        else getattr(item, "text", "")
                    )
                    text_blocks.append({"type": "text", "text": item_text})

        if text_blocks:
            # Anthropic accepts a list of content blocks (text, images, etc.).
            return {"role": role, "content": text_blocks}

    # Safe fallback when content format is unexpected.
    return {"role": role, "content": ""}


def convert_prompt_messages_to_message_params(
    prompt_messages: List[PromptMessage],
) -> List[MessageParam]:
    # Batch converter used after fetching prompt templates from MCP.
    return [
        convert_prompt_message_to_message_param(msg) for msg in prompt_messages
    ]
