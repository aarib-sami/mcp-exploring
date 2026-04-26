from anthropic import Anthropic
from anthropic.types import Message


class Claude:
    # Thin wrapper around the Anthropic SDK to keep model calls/message helpers
    # in one place instead of scattering SDK details across the app.
    def __init__(self, model: str):
        # Reads API key from environment (ANTHROPIC_API_KEY).
        self.client = Anthropic()
        # Model id is configured by main.py (from .env).
        self.model = model

    def add_user_message(self, messages: list, message):
        # Normalizes either an Anthropic Message object or plain content
        # into this app's conversation history format.
        user_message = {
            "role": "user",
            "content": message.content
            if isinstance(message, Message)
            else message,
        }
        messages.append(user_message)

    def add_assistant_message(self, messages: list, message):
        # Same normalization helper for assistant-role messages.
        assistant_message = {
            "role": "assistant",
            "content": message.content
            if isinstance(message, Message)
            else message,
        }
        messages.append(assistant_message)

    def text_from_message(self, message: Message):
        # Extract only text blocks; ignores non-text blocks like tool_use.
        return "\n".join(
            [block.text for block in message.content if block.type == "text"]
        )

    def chat(
        self,
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=[],
        tools=None,
        thinking=False,
        thinking_budget=1024,
    ) -> Message:
        # Shared default request params for every model call.
        params = {
            "model": self.model,
            "max_tokens": 8000,
            "messages": messages,
            "temperature": temperature,
            "stop_sequences": stop_sequences,
        }

        if thinking:
            # Enables Anthropic "thinking" mode with an explicit token budget.
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }

        if tools:
            # Tool schemas exposed to the model for tool-use planning/calls.
            params["tools"] = tools

        if system:
            # Optional system instruction layer.
            params["system"] = system

        # Blocking SDK call that returns one assistant message.
        message = self.client.messages.create(**params)
        return message
