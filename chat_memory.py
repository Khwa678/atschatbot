# chat_memory.py
"""
This module manages short-term chat memory using a sliding window.
It stores recent user and bot messages and builds context prompts for the model.
"""

from collections import deque

class ChatMemory:
    """
    Manages a sliding window of conversation history.

    Each turn is stored as (role, text), where role ∈ {"user", "bot"}.
    """

    def __init__(self, max_turns: int = 3):
        """
        Initialize memory buffer.

        Args:
            max_turns (int): Number of user-bot pairs to remember.
        """
        # Each turn has a user and a bot message → 2 entries per turn
        self.buffer = deque(maxlen=max_turns * 2)
        self.max_turns = max_turns

    def add_user(self, text: str):
        """Add a user message to memory."""
        self.buffer.append(("user", text.strip()))

    def add_bot(self, text: str):
        """Add a bot message to memory."""
        self.buffer.append(("bot", text.strip()))

    def get_prompt_context(self, system_prompt: str = None) -> str:
        """
        Build the text prompt for the model, including recent history.

        Args:
            system_prompt (str): Optional instruction or role description for the model.

        Returns:
            str: Combined conversation history text for generation.
        """
        lines = []
        if system_prompt:
            lines.append(system_prompt.strip())

        for role, text in self.buffer:
            if role == "user":
                lines.append(f"User: {text}")
            else:
                lines.append(f"Bot: {text}")

        # Return all lines joined by newlines + a new "Bot:" cue for next response
        return "\n".join(lines)

    def clear(self):
        """Clear the memory buffer."""
        self.buffer.clear()

    def __repr__(self):
        """Representation for debugging."""
        return f"ChatMemory(max_turns={self.max_turns}, buffer={list(self.buffer)})"


# Example usage
if __name__ == "__main__":
    memory = ChatMemory(max_turns=2)
    memory.add_user("Hi, who are you?")
    memory.add_bot("I'm a chatbot.")
    memory.add_user("What can you do?")
    print(memory.get_prompt_context(system_prompt="You are a helpful AI assistant."))
