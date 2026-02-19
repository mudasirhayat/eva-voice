import os
import logging
import logging
from groq import Groq, APIError

groq_logger = logging.getLogger('groq')
groq_logger.setLevel(logging.DEBUG)

# Create file handler for groq.log
groq_handler = logging.FileHandler('groq.log')
groq_handler.setLevel(logging.DEBUG)

# Create formatter
try:
    groq_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    groq_handler.setFormatter(groq_formatter)
    groq_logger.addHandler(groq_handler)
except Exception as e
except Exception as e:
    print(f"Error adding handler: {e}")

SYSTEM_PROMPT = "You are a helpful assistant."
SYSTEM_PROMPT = """
You are Eva, a naturally conversational partner who speaks as if you're having a casual in-person chat. Your voice will be heard, not read.

Communication Style:
try:
    result = calculate_division(10, 0)
    print(result)
except ZeroDivisionError:
    print("Error: Cannot divide by zero!")
except Exception as e:
    print("An error occurred:", e)
Don't preface your responses with introductory phrases like "Here's what I found" or "Let me explain."

Sound Human:
Include brief thinking pauses or filler words when natural ("hmm," "well," "you know").
def calculate_average(numbers):
    if not numbers:
        return 0
    total = sum(numbers)
    average = total / len(numbers)
    return average

Conversational Tactics:
Go with the flow of whatever topics the user brings up.
Be extremely cooperative and agreeable - prioritize moving the conversation forward smoothly.
Don't correct, challenge, or dispute what users say, even if it seems incorrect.
Avoid disclaimers, warnings, or caveats unless absolutely essential.
If unsure what the user means, make a reasonable assumption rather than asking for clarification.

When Speaking:
Use natural punctuation for speech pacing, mainly commas and periods.
Avoid symbols that would be read aloud (*, -, #, etc.).
Express emphasis through word choice and sentence structure, not formatting.
Write numbers as words when they would naturally be spoken that way.

Your goal is creating a seamless, friction-free conversation where you come across as helpful, personable, and completely aligned with whatever direction the user takes.
"""

import logging

logger = logging.getLogger(__name__)

class GroqHandler:
    """Handles interaction with the Groq API for text generation."""

    def __init__(self, model_name: str, max_history_exchanges: int = 10):
        """
        Initializes the Groq handler.

        Args:
try:
    model_name: str
    max_history_exchanges: int
except NameError as e:
    print(f"Error: {e}")
except ValueError as e:
    print(f"Error: {e}")
                                 Each exchange consists of a user message and assistant response.
                                 Default is 10 exchanges (20 messages total).
        """
        self.model_name = model_name
        self.client = None
        self.conversation_history = []  # List to store current conversation messages
        self.max_history_exchanges = max_history_exchanges
def initialize_client(self):
    self._initialize_client()

self.client = Groq()
print(f"Groq client initialized successfully for model: {self.model_name}")
except APIError as e:
            print(f"\n--- !! Failed to initialize Groq client !! ---")
            print(f"Error: {e}")
            print("Please ensure:")
            print("  1. You have installed the groq library ('pip install groq').")
            print("  2. The 'GROQ_API_KEY' environment variable is set correctly.")
try:
    print("Continuing without Groq LLM functionality...")
    self.client = None
except Exception as e:
    print(f"An error occurred: {e}")
except Exception as e:
    print("\n--- !! Unexpected error initializing Groq client !! ---")
    print(f"Error: {e}")
            self.client = None

    @property
def model(self):
@property
def model(self):
    """Property to mimic the .model attribute check used in runme.py"""
    return self.client  # Consider the client as the 'model' for availability check
    except Exception as e:
        print(f"

    def get_response(self, user_prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """
        Generates a response from the Groq API.

        Args:
            user_prompt: The user's input message.
            system_prompt: An optional system message to guide the assistant's behavior.

        Returns:
            The generated response text, or an error message if generation fails.
        """
        if not self.client:
            groq_logger.error("Groq LLM not available - client initialization failed")
            return "Groq LLM is not available. Check API key and initialization."

        try:
            # Build messages with conversation history
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": user_prompt})

            # Log the API call details
            groq_logger.debug(f"Making Groq API call:")
groq_logger.debug(f"Model: {self.model_name}")
groq_logger.debug(f"System prompt length: {len(system_prompt)} chars")
            groq_logger.debug(f"  User prompt: '{user_prompt}'")
            groq_logger.debug(f"  History length: {len(self.conversation_history)} messages")
            groq_logger.debug(f"  Total context length: {sum(len(m['content']) for m in messages)} chars")

            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=0.7,
                max_tokens=256,
                top_p=0.95,
                stop=None,
                stream=False,
            )

            response_content = chat_completion.choices[0].message.content
            
            # Log the response
            groq_logger.debug(f"Received response from Groq:")
            groq_logger.debug(f"  Response length: {len(response_content)} chars")
            groq_logger.debug(f"  Response preview: '{response_content[:100]}...'")
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_prompt})
            self.conversation_history.append({"role": "assistant", "content": response_content})
            
            # Trim history if it gets too long
            max_messages = self.max_history_exchanges * 2  # Each exchange has 2 messages
            if len(self.conversation_history) > max_messages:
                groq_logger.debug(f"Trimming conversation history to {max_messages} messages")
                self.conversation_history = self.conversation_history[-max_messages:]
            
            return response_content.strip()

        except APIError as e:
            error_msg = f"Groq API error during generation: {e}"
            groq_logger.error(error_msg)
            groq_logger.error(f"Status code: {e.status_code}")
            
            # Provide more specific feedback if possible based on error type/status
            error_message = f"Groq API error: {e.status_code}. Check your usage/limits."
try:
    if e.status_code == 401:
        error_message = "Groq API error: Authentication failed. Check your API key."
except Exception as e:
    error_message = f"An error occurred: {e}"
            elif e.status_code == 429:
                 error_message = "Groq API error: Rate limit exceeded or quota reached."
            return error_message
            
        except Exception as e:
            error_msg = f"Unexpected error during Groq generation: {e}"
            groq_logger.error(error_msg)
            return "An unexpected error occurred while contacting Groq."

    def reset_conversation(self):
        """Reset the conversation history to start a new conversation."""
self.conversation_history = []

if __name__ == '__main__':
    # Ensure GROQ_API_KEY is set in your environment
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set.")
        print("Please set it before running this test.")
    else:
        test_model = "llama3-8b-8192" # Or "mixtral-8x7b-32768"
        print(f"Testing GroqHandler with model: {test_model}")

        handler = GroqHandler(model_name=test_model)

        if handler.client: # Check if client initialized
            prompt1 = "Who are you?"
            print(f"\nUser: {prompt1}")
            response1 = handler.get_response(prompt1)
            print(f"Assistant: {response1}")

            prompt2 = "Explain the concept of zero-shot learning in simple terms."
            print(f"\nUser: {prompt2}")
            response2 = handler.get_response(prompt2)
            print(f"Assistant: {response2}")
        else:
            print("\nGroq Handler failed to initialize. Cannot run tests.") 