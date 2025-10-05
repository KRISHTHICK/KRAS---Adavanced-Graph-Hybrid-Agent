import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jVector
from langchain_core.documents import Document

# --- 0. Environment Setup ---
load_dotenv()

# Gemini Keys
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Neo4j Details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# File Path (Ensure 'Annual_Report.pdf' is in your 'data_sources' folder)
PDF_FILE_PATH = "data_sources/Annual_Report.pdf"
VECTOR_INDEX_NAME = "pdf_vector_index"
NODE_LABEL = "Chunk" # The name of the node that will store the vector data

def load_and_index_documents():
    print(f"1. Loading and parsing PDF: {PDF_FILE_PATH}")
    # Use PDFPlumberLoader for better table/layout extraction
    try:
        loader = PDFPlumberLoader(PDF_FILE_PATH)
        documents = loader.load()
    except Exception as e:
        print(f"🔴 ERROR loading PDF: {e}. Ensure the file exists and is readable.")
        return

    # 2. Chunking Documents
    # RecursiveCharacterTextSplitter maintains semantic coherence (paragraphs/sentences)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    # Split the loaded document objects
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Document split into {len(chunks)} chunks.")

    # 3. Initialize Embeddings and Vector Store
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=GOOGLE_KEY
    )
    print(f"✅ Initialized Gemini Embedding Model: {EMBEDDING_MODEL}")

    # 4. Create Index and Push to Neo4j
    print(f"4. Creating Vector Index '{VECTOR_INDEX_NAME}' and loading chunks into Neo4j...")
    
    try:
        # Neo4jVector automatically creates the vector index and populates the nodes/vectors
        db = Neo4jVector.from_documents(
            chunks, 
            embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name=VECTOR_INDEX_NAME,
            node_label=NODE_LABEL
        )
        print("----------------------------------------------------------------")
        print(f"🚀 PHASE 1 COMPLETE! {len(chunks)} Chunks and vectors indexed successfully in Neo4j.")
        print(f"Vector Index Name: {VECTOR_INDEX_NAME}")
        print("----------------------------------------------------------------")
        
        # Optional: Test a simple semantic search to confirm retrieval works
        query = "What are the most recent financial highlights mentioned in the report?"
        retrieved_docs = db.similarity_search(query, k=1)
        print("\n--- Verification Query Test ---")
        print(f"Found related text: {retrieved_docs[0].page_content[:200]}...")
        
    except Exception as e:
        print(f"🔴 ERROR during Neo4j indexing: {e}. Check your URI/Credentials/Permissions.")

if __name__ == "__main__":
    load_and_index_documents()