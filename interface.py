# interface.py
"""
Main command-line interface for the local Hugging Face chatbot.
Connects the model loader and memory buffer to create an interactive chatbot.
"""

import argparse
from model_loader import ModelLoader
from chat_memory import ChatMemory
import textwrap

def main():
    # ----- CLI argument setup -----
    parser = argparse.ArgumentParser(description="Local CLI Chatbot using Hugging Face")
    parser.add_argument("--model", type=str, default="distilgpt2",
                        help="Hugging Face model name (default: distilgpt2)")
    parser.add_argument("--max-turns", type=int, default=3,
                        help="Number of recent turns to remember (default: 3)")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum tokens to generate in each response (default: 100)")
    args = parser.parse_args()

    # ----- Load model -----
    print(f"🔹 Loading model '{args.model}' ... This may take a few seconds.\n")
    model = ModelLoader(model_name=args.model)
    memory = ChatMemory(max_turns=args.max_turns)

    system_prompt = "You are a helpful AI assistant. Keep your responses short and clear."

    print("🤖 Chatbot is ready!")
    print("Type your message and press Enter.")
    print("Use '/exit' to quit or '/clear' to reset memory.\n")

    # ----- Chat Loop -----
    while True:
        try:
            user_input = input("User: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chatbot. Goodbye!")
            break

        if not user_input:
            continue

        # Exit command
        if user_input.lower() == "/exit":
            print("Exiting chatbot. Goodbye!")
            break

        # Clear memory
        if user_input.lower() == "/clear":
            memory.clear()
            print("[Memory cleared]\n")
            continue

        # Add user message to memory
        memory.add_user(user_input)

        # Build prompt with recent conversation
        prompt_context = memory.get_prompt_context(system_prompt=system_prompt)
        prompt = prompt_context + "\nBot:"

        # Generate response
        raw_output = model.generate(
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=0.7,
            top_p=0.9
        )

        # Extract the bot's new reply (after "Bot:")
        if raw_output.startswith(prompt):
            bot_reply = raw_output[len(prompt):].strip()
        else:
            bot_reply = raw_output.strip().splitlines()[-1]

        # Format reply neatly
        bot_reply = textwrap.fill(bot_reply, width=90)

        print(f"Bot: {bot_reply}\n")

        # Add bot reply to memory
        memory.add_bot(bot_reply)


if __name__ == "__main__":
    main()
