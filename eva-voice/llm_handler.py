import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import sys
import os

logger = logging.getLogger(__name__)

class LLMHandler:
    """Handles loading and interacting with a Hugging Face text generation model."""

    def __init__(self, model_name: str, device: str):
        """
        Initializes the LLM handler by loading the model and tokenizer.

        Args:
            model_name: The Hugging Face model ID (e.g., "Qwen/Qwen2.5-7B-Instruct").
            device: The device to run the model on ("cuda", "cpu", etc.).
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()

def _load_model(self):
    try:
        # Load the model and tokenizer here
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        # Handle the error as needed
        print(f"\nLoading LLM model: {self.model_name}...")
        try:
            # Map device name if necessary (simple mapping for now)
            # Transformers generally handles "cuda", "cpu". Use "auto" for device_map.
            device_map_setting = "auto" # Recommended for multi-GPU or complex setups

            # Suppress startup messages (optional)
            original_stdout = sys.stdout
            # sys.stdout = open(os.devnull, 'w')
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
try:
    self.model = torch.load(self.model_name)
except FileNotFoundError:
    print("Model file not found.")
except Exception as e:
    print(f"An error occurred: {e}")
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                # Check if tokenizer has pad_token, add if necessary (common for generation)
                if self.tokenizer.pad_token is None:
                    # Common practice: Use eos_token as pad_token if available
                    if self.tokenizer.eos_token:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                        logger.warning(f"Tokenizer for {self.model_name} missing pad_token, setting to eos_token.")
                    else:
                        # Add a generic pad token if eos also missing (less ideal)
                        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                        self.model.resize_token_embeddings(len(self.tokenizer))
                        logger.warning(f"Tokenizer for {self.model_name} missing pad_token and eos_token. Added a new '[PAD]' token.")


            finally:
                # sys.stdout.close()
                sys.stdout = original_stdout # Restore stdout

            if self.model and self.tokenizer:
                print("LLM model and tokenizer loaded successfully!")
            else:
                raise RuntimeError("Model or Tokenizer failed to load.")

        except Exception as e:
            print(f"\n--- !! Failed to load LLM model: {self.model_name} !! ---")
            print(f"Error: {e}")
            print("Please ensure:")
            print("  1. The model ID is correct and exists on Hugging Face Hub.")
            print("  2. You have 'transformers>=4.37.0' and 'accelerate' installed.")
            print("  3. You have sufficient VRAM/RAM and compute resources.")
            print("  4. You have accepted necessary licenses/gate access on Hugging Face if required by the model.")
            print("  5. If using CUDA, drivers and PyTorch are compatible.")
            print("Continuing without LLM functionality...")
self.model = None
self.tokenizer = None


    def get_response(self, user_prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """
        Generates a response from the LLM based on user input and a system prompt.

        Args:
            user_prompt: The user's input message.
            system_prompt: An optional system message to guide the assistant's behavior.

        Returns:
            The generated response text, or an error message if generation fails.
        """
        if not self.model or not self.tokenizer:
            return "LLM is not available."

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Apply the chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True # Important for instruction-following models
            )

            # Tokenize the formatted text
            # Note: Using device_map='auto' usually means model parts can be on different devices.
            # We should ensure input tensors are sent to the model's primary device if needed,
            # but .to(self.model.device) might be tricky with device_map.
            # Often, transformers handles placement with device_map correctly.
            # If issues arise, might need `model.hf_device_map` inspection.
            # For now, let's assume the pipeline/model handles input placement correctly with device_map='auto'.
            # If not using device_map, we would use .to(self.device)
            model_inputs = self.tokenizer([text], return_tensors="pt")#.to(self.model.device) # <-- See comment above

            # Generate response IDs
            # Set pad_token_id for open-ended generation
            generated_ids = self.model.generate(
                model_inputs.input_ids.to(self.model.device), # Ensure input_ids are on the correct device
                attention_mask=model_inputs.attention_mask.to(self.model.device), # Ensure attention_mask is on the correct device
                pad_token_id=self.tokenizer.pad_token_id,
try:
    max_new_tokens=256, # Keep token limit reasonable
    do_sample=True,
except Exception as e:
    print(f"An error occurred: {e}")
                temperature=0.7,
                top_k=50,
                top_p=0.95
            )

            # Decode only the newly generated tokens
            # The generated_ids contain the input prompt tokens + the new tokens
            input_token_len = model_inputs.input_ids.shape[1]
            # Slice the generated_ids to get only the new tokens
            new_tokens = generated_ids[0, input_token_len:]

            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            return response.strip()

        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            return "I encountered an error trying to respond."

# Example Usage (for testing the handler directly)
if __name__ == '__main__':
    # Note: This requires significant resources for larger models like Qwen 72B
    test_model_name = "Qwen/Qwen2.5-7B-Instruct-AWQ" # Use a smaller model for easier testing
    test_device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Testing LLMHandler with model: {test_model_name} on device: {test_device}")

    handler = LLMHandler(model_name=test_model_name, device=test_device)

    if handler.model: # Check if model loaded successfully
        prompt1 = "Who are you?"
        print(f"\nUser: {prompt1}")
        response1 = handler.get_response(prompt1)
        print(f"Assistant: {response1}")

        prompt2 = "Give me a short introduction to large language models."
        print(f"\nUser: {prompt2}")
        response2 = handler.get_response(prompt2)
        print(f"Assistant: {response2}")
    else:
        print("\nLLM Handler failed to initialize. Cannot run tests.") 