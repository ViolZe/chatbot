import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- Securely Load API Key from .env file ---
# This function loads the environment variables from a .env file.
# It's the first thing we do to ensure the API key is available.
load_dotenv()


# --- Configuration ---
# Now, we configure the API by safely reading the key from the environment.
try:
    # Fetches the API key loaded from the .env file.
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    # This error will be raised if the key is not found in the .env file
    # or if the .env file is missing.
    print("🔴 Error: GOOGLE_API_KEY not found.")
    print("👉 Please create a file named `.env` in the same directory.")
    print("   In that file, add the line: GOOGLE_API_KEY='YOUR_KEY_HERE'")
    exit()


# Model Configuration
generation_config = {
    "temperature": 0.8,
    "top_p": 1.0,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# Safety Settings
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# --- Main Chatbot Logic ---

def run_chatbot(persona: str):
    """
    The main function to run the chatbot application with a given persona.
    """
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config=generation_config,
        safety_settings=safety_settings,
        system_instruction=persona
    )

    chat = model.start_chat(history=[])

    print("-" * 60)
    print(f"🤖 Chatbot initialized with persona: '{persona}'")
    print("   You can start chatting now. Type 'quit' or 'exit' to end.")
    print("-" * 60)

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit"]:
            print(f"\n🤖 Goodbye! It was fun playing '{persona}'.")
            break

        if not user_input:
            continue

        try:
            response = chat.send_message(user_input, stream=True)

            print("\nGemini: ", end="", flush=True)
            for chunk in response:
                print(chunk.text, end="", flush=True)
            print("\n")

        except Exception as e:
            print(f"🔴 An error occurred: {e}")
            break

# --- Entry Point ---
if __name__ == "__main__":
    print("🚀 Let's set up your chatbot!")
    persona_input = input("First, what should the chatbot be? (e.g., 'a pirate', 'a helpful fitness coach')\n> ").strip()

    if not persona_input:
        persona_input = "A helpful and friendly AI assistant."
        print(f"No persona given. Using default: '{persona_input}'")

    run_chatbot(persona_input)