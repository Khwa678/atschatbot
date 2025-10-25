# model_loader.py
"""
This module loads a Hugging Face text-generation model and tokenizer.
It provides a class (ModelLoader) that wraps the model using the
Transformers pipeline API for simple text generation.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class ModelLoader:
    """
    Handles model and tokenizer loading, and provides a simple text generation method.
    """

    def __init__(self, model_name: str = "distilgpt2", device: str = None):
        """
        Initialize the model and tokenizer.

        Args:
            model_name (str): Hugging Face model ID (default = "distilgpt2").
            device (str): "cpu" or "cuda". If None, selects automatically.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = 0 if device == "cuda" else -1  # for pipeline

        print(f"🔹 Loading model '{model_name}' on {device.upper()}...")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Create a Hugging Face pipeline for text generation
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        print("✅ Model and tokenizer loaded successfully!\n")

    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate a response based on the input prompt.

        Args:
            prompt (str): Text prompt to feed the model.
            max_new_tokens (int): Max number of tokens to generate.
            temperature (float): Sampling temperature for randomness.
            top_p (float): Nucleus sampling parameter.

        Returns:
            str: The full generated text (including the prompt).
        """
        outputs = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        return outputs[0]["generated_text"]


# Example usage (for quick testing only)
if __name__ == "__main__":
    chatbot_model = ModelLoader("distilgpt2")
    prompt = "User: Hello!\nBot:"
    print(chatbot_model.generate(prompt))
