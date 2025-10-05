# # import os
# # from dotenv import load_dotenv
# # from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# # load_dotenv() # Load the secrets

# # # Retrieve environment variables
# # GEMINI_KEY = os.getenv("GEMINI_API_KEY")
# # LLM_MODEL = os.getenv("LLM_MODEL")
# # EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# # # 1. LLM Test (Chat Model for Generation)
# # print("--- LLM Test ---")
# # try:
# #     # Pass the key explicitly or rely on the OS environment variable
# #     llm = ChatGoogleGenerativeAI(model=LLM_MODEL, api_key=GEMINI_KEY)
# #     response = llm.invoke("What is a Cypher query in 5 words?")
# #     print(f"✅ LLM Model ({LLM_MODEL}) response: {response.content.strip()}")
# # except Exception as e:
# #     print(f"🔴 LLM Test FAILED: Check your API Key and Model Name. Error: {e}")

# # # 2. Embeddings Test
# # print("\n--- Embeddings Test ---")
# # try:
# #     embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=GEMINI_KEY)
# #     vector = embeddings.embed_query("financial statement analysis")
# #     print(f"✅ Embedding Model ({EMBEDDING_MODEL}) Success: Vector length is {len(vector)}")
# # except Exception as e:
# #     print(f"🔴 Embeddings Test FAILED: Check your API Key and Model Name. Error: {e}")

# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# load_dotenv() # Load the secrets

# # --- Retrieve the universal Google API Key ---
# # Use GOOGLE_API_KEY, as it is the standard for the underlying libraries.
# GOOGLE_KEY = os.getenv("GOOGLE_API_KEY") 
# LLM_MODEL = os.getenv("LLM_MODEL")
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# if not GOOGLE_KEY:
#     print("FATAL ERROR: GOOGLE_API_KEY is not set in the .env file.")
#     exit()

# # 1. LLM Test (Chat Model for Generation)
# print("--- LLM Test ---")
# try:
#     # Pass the key explicitly to the constructor using the 'api_key' argument
#     llm = ChatGoogleGenerativeAI(model=LLM_MODEL, api_key=GOOGLE_KEY) 
#     response = llm.invoke("What is a Cypher query in 5 words?")
#     print(f"✅ LLM Model ({LLM_MODEL}) response: {response.content.strip()}")
# except Exception as e:
#     print(f"🔴 LLM Test FAILED: Error: {e}")

# # 2. Embeddings Test (for Vector Search)
# print("\n--- Embeddings Test ---")
# try:
#     # Pass the key explicitly to the constructor using the 'api_key' argument
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model=EMBEDDING_MODEL, 
#         api_key=GOOGLE_KEY # <<< FINAL CRITICAL FIX
#     )
#     vector = embeddings.embed_query("financial statement analysis")
#     print(f"✅ Embedding Model ({EMBEDDING_MODEL}) Success: Vector length is {len(vector)}")
# except Exception as e:
#     print(f"🔴 Embeddings Test FAILED: Error: {e}")

import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from urllib.parse import urlparse

load_dotenv()

# --- Configuration ---
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash-exp")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

if not GOOGLE_KEY:
    print("🔴 FATAL ERROR: GOOGLE_API_KEY is not set in the .env file.")
    exit()

# --- Helper Functions ---

def is_url(path):
    """Check if the path is a URL"""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except:
        return False

def download_pdf(url, save_path="temp_downloaded.pdf"):
    """Download PDF from URL"""
    try:
        print(f"📥 Downloading PDF from URL: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Downloaded to: {save_path}")
        return save_path
    except Exception as e:
        print(f"🔴 Error downloading PDF: {e}")
        return None

def load_pdf(source):
    """
    Load PDF from file path or URL
    
    Args:
        source: File path (str) or URL (str)
    
    Returns:
        List of documents or None if failed
    """
    temp_file = None
    
    try:
        # Check if source is a URL
        if is_url(source):
            temp_file = download_pdf(source)
            if not temp_file:
                return None
            file_path = temp_file
        else:
            # It's a local file path
            file_path = source
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"🔴 ERROR: File not found: {file_path}")
                print(f"💡 Current directory: {os.getcwd()}")
                print(f"💡 Looking for: {os.path.abspath(file_path)}")
                return None
        
        # Load the PDF
        print(f"📄 Loading PDF: {file_path}")
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        
        print(f"✅ Successfully loaded {len(documents)} page(s)")
        
        # Preview first page
        if documents:
            preview = documents[0].page_content[:200]
            print(f"📖 First page preview: {preview}...")
        
        return documents
        
    except Exception as e:
        print(f"🔴 ERROR loading PDF: {e}")
        return None
    
    finally:
        # Clean up temporary file if downloaded
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f"🧹 Cleaned up temporary file: {temp_file}")
            except:
                pass

# --- Main Script ---

def main():
    print("=" * 60)
    print("📚 FLEXIBLE PDF READER - File or URL")
    print("=" * 60)
    
    # Example 1: Try local file
    print("\n--- Test 1: Local File ---")
    local_file = "data_sources/Annual_Report.pdf"
    
    # Create directory if it doesn't exist
    os.makedirs("data_sources", exist_ok=True)
    
    docs = load_pdf(local_file)
    
    if not docs:
        print("\n💡 TIP: Place your PDF in the 'data_sources' folder")
        print(f"💡 Expected location: {os.path.abspath(local_file)}")
    
    # Example 2: Try URL (sample PDF)
    print("\n--- Test 2: PDF from URL ---")
    sample_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    docs_from_url = load_pdf(sample_url)
    
    # Example 3: Interactive mode
    print("\n" + "=" * 60)
    print("🎯 INTERACTIVE MODE")
    print("=" * 60)
    
    user_input = input("\n📎 Enter PDF file path or URL (or 'skip' to exit): ").strip()
    
    if user_input and user_input.lower() != 'skip':
        docs = load_pdf(user_input)
        
        if docs:
            print(f"\n✅ SUCCESS! Loaded {len(docs)} pages")
            
            # Optional: Test with LLM
            test_llm = input("\n🤖 Want to analyze with Gemini? (y/n): ").strip().lower()
            
            if test_llm == 'y':
                try:
                    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, api_key=GOOGLE_KEY)
                    
                    # Summarize first page
                    first_page = docs[0].page_content[:1000]  # Limit to 1000 chars
                    prompt = f"Summarize this text in 2-3 sentences:\n\n{first_page}"
                    
                    response = llm.invoke(prompt)
                    print(f"\n📝 AI Summary:\n{response.content}")
                    
                except Exception as e:
                    print(f"🔴 LLM Error: {e}")

if __name__ == "__main__":
    main()