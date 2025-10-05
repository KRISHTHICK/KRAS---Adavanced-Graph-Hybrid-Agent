import os
from dotenv import load_dotenv

# This line loads the variables from the .env file
load_dotenv()

print("--- Environment Check ---")
print(f"Gemini API Key Loaded: {bool(os.getenv('GEMINI_API_KEY'))}")
print(f"Neo4j URI Loaded: {os.getenv('NEO4J_URI')}")
print("-------------------------")