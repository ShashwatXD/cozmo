"""
Chat use-case (simple, no tools) + conversation memory.

What: multi-turn chat with sliding-window history.
Why: follow-up questions need prior context.
Layer: app (ViewModel).
Flutter: ChatCubit holding List<Message>.
"""

from collections.abc import Iterator

from cozmo.domain.completion import CompletionResult
from cozmo.domain.memory import ConversationMemory
from cozmo.domain.messages import Message, Role
from cozmo.domain.ports import LLMClient
from cozmo.prompts.loader import load_system_prompt


class ChatUseCase:
    """Flutter: ChatCubit — depends on LLMClient port, not openai."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        memory: ConversationMemory | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> None:
        self._llm = llm
        self._memory = memory if memory is not None else ConversationMemory()
        self._system_prompt = system_prompt or load_system_prompt("default")
        self._temperature = temperature

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    def run(self, user_text: str, *, json_mode: bool = False) -> CompletionResult:
        messages = [
            *self._memory.for_prompt(self._system_prompt),
            Message(role=Role.USER, content=user_text),
        ]
        result = self._llm.complete(
            messages,
            temperature=self._temperature,
            json_mode=json_mode,
            tools=None,
        )
        self._memory.add(Message(role=Role.USER, content=user_text))
        self._memory.add(Message(role=Role.ASSISTANT, content=result.content))
        return result

    def stream(self, user_text: str) -> Iterator[str]:
        messages = [
            *self._memory.for_prompt(self._system_prompt),
            Message(role=Role.USER, content=user_text),
        ]
        chunks: list[str] = []
        for chunk in self._llm.stream(messages, temperature=self._temperature):
            chunks.append(chunk)
            yield chunk
        text = "".join(chunks)
        self._memory.add(Message(role=Role.USER, content=user_text))
        self._memory.add(Message(role=Role.ASSISTANT, content=text))
