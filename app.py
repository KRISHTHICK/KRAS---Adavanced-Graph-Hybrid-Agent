# # import os
# # import streamlit as st
# # from dotenv import load_dotenv
# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain_core.output_parsers import StrOutputParser
# # from langchain_core.runnables import RunnablePassthrough, RunnableParallel
# # from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# # from langchain_community.graphs import Neo4jGraph
# # from langchain_neo4j import Neo4jVector # For Vector Search
# # from langchain_neo4j.chains import GraphCypherQAChain # For Graph Search

# import os
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough, RunnableParallel
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.graphs import Neo4jGraph
# from langchain_neo4j import Neo4jVector  # For Vector Search
# from langchain.chains import GraphCypherQAChain  # ← FIXED: Correct import path
# # --- 0. Setup and Initialization ---
# load_dotenv()

# # Retrieve credentials and settings from .env
# GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
# NEO4J_URI = os.getenv("NEO4J_URI")
# NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
# LLM_MODEL = os.getenv("LLM_MODEL")
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# # --- Initialize Core Components (LLM, Embeddings, Neo4j) ---

# # Gemini Chat Model (for generation and Cypher query translation)
# llm = ChatGoogleGenerativeAI(model=LLM_MODEL, api_key=GOOGLE_KEY, temperature=0)

# # Gemini Embeddings Model (for vector search)
# embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=GOOGLE_KEY)

# # Neo4j Graph Connection (Used for Cypher QA Chain)
# neo4j_graph = Neo4jGraph(
#     url=NEO4J_URI,
#     username=NEO4J_USERNAME,
#     password=NEO4J_PASSWORD
# )

# # --- 1. Graph Search (Structured Data Retrieval) ---

# # This chain translates the question into Cypher, runs it, and generates an answer.
# graph_qa_chain = GraphCypherQAChain.from_llm(
#     llm=llm,
#     graph=neo4j_graph,
#     verbose=True,
#     return_intermediate_steps=True,
#     allow_dangerous_requests=True  # ← Add this line only for demo
# )

# # --- 2. Vector Search (Unstructured Data Retrieval) ---

# # This retriever searches the PDF chunks you loaded in Phase 1 (Node: Chunk)
# vector_retriever = Neo4jVector.from_existing_index(
#     embeddings,
#     url=NEO4J_URI,
#     username=NEO4J_USERNAME,
#     password=NEO4J_PASSWORD,
#     index_name="pdf_vector_index",
#     node_label="Chunk", # This links to the nodes you created in extract_pdf_vectors.py
#     text_node_property="text", 
#     embedding_node_property="embedding",
#     retrieval_query="""
#         RETURN node.text AS text, score, {source: node.source_file, type: "PDF Chunk"} AS metadata
#     """
# ).as_retriever(search_kwargs={"k": 3})


# # --- 3. Hybrid RAG (The Intelligence Agent) ---

# def run_hybrid_query(question):
#     # A simple router: try the graph first for structured, precise queries.
#     if any(keyword in question.lower() for keyword in ["value", "metric", "ratio", "equity", "debt", "rate"]):
#         st.info("Agent Action: Executing **Graph Search (Cypher)** for Structured Data.")
        
#         # Run the Graph Chain
#         graph_result = graph_qa_chain.invoke({"query": question})
#         cypher_query = graph_result['intermediate_steps'][0]['query']
#         st.code(cypher_query, language='cypher')
#         st.caption("Generated Cypher Query:")
#         return graph_result['result'], cypher_query
    
#     else:
#         st.info("Agent Action: Executing **Vector Search** for Unstructured Context.")
        
#         # Run the Vector Retriever
#         vector_docs = vector_retriever.invoke(question)
#         context = "\n\n".join([doc.page_content for doc in vector_docs])
        
#         # Define the final RAG chain prompt
#         rag_prompt = ChatPromptTemplate.from_messages([
#             ("system", "You are a professional financial analyst. Use ONLY the following contextual information to answer the question. Do not hallucinate. \n\nCONTEXT:\n{context}"),
#             ("human", "Question: {question}")
#         ])

#         # Combine components using LangChain Expression Language (LCEL)
#         rag_chain = (
#             RunnableParallel(
#                 context=lambda x: context,
#                 question=RunnablePassthrough()
#             ) | rag_prompt | llm | StrOutputParser()
#         )
        
#         vector_answer = rag_chain.invoke({"question": question})
#         return vector_answer, None

# # --- STREAMLIT UI ---
# st.set_page_config(page_title="Advanced Graph RAG Demo", layout="wide")
# st.title("🧠 Advanced Graph RAG Demo (Gemini + Neo4j)")
# st.caption("A Hybrid AI Agent combining Structured Data (Excel) and Unstructured Context (PDF)")

# # --- Chat Interface ---
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#         if "details" in message:
#             st.code(message["details"], language='cypher')
#             st.caption("Cypher Query Used")

# if prompt := st.chat_input("Ask a question about the company's financial data or annual report..."):
#     # Store user message
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.spinner("Thinking..."):
#         # Run the Hybrid Agent
#         final_answer, cypher_query = run_hybrid_query(prompt)

#     # Store AI response
#     ai_message = {"role": "assistant", "content": final_answer}
#     if cypher_query:
#         ai_message["details"] = cypher_query
    
#     st.session_state.messages.append(ai_message)
    
#     # Display the final response
#     with st.chat_message("assistant"):
#         st.markdown(final_answer)
#         if cypher_query:
#             st.code(cypher_query, language='cypher')
#             st.caption("Cypher Query Used")

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_neo4j import Neo4jVector
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

# --- 0. Setup and Initialization ---
load_dotenv()

# Retrieve credentials and settings from .env
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Validate credentials
if not all([GOOGLE_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    st.error("Missing environment variables. Please check your .env file.")
    st.stop()

# --- Initialize Core Components (LLM, Embeddings, Neo4j) ---

# Gemini Chat Model (for generation and Cypher query translation)
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, api_key=GOOGLE_KEY, temperature=0)

# Gemini Embeddings Model (for vector search)
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=GOOGLE_KEY)

# Neo4j Graph Connection (Used for Cypher QA Chain)
neo4j_graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# Refresh schema to help LLM understand the graph structure
neo4j_graph.refresh_schema()

# --- 1. Graph Search (Structured Data Retrieval) ---

# Custom Cypher prompt with better examples
cypher_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", """Task: Generate Cypher statement to query a graph database.
Instructions:
- Use only the provided relationship types and properties in the schema
- Do not use any other relationship types or properties that are not provided
- Write only ONE Cypher query with ONE RETURN clause at the end
- Be precise with property names and node labels

Schema:
{schema}

Examples:
Question: What is the Equity value for Zenalyst Inc.?
Cypher: MATCH (c:Company {{name: 'Zenalyst Inc.'}})-[:HAS_METRIC]->(m:Metric {{name: 'Equity'}}) RETURN m.value AS equity_value

Question: Show all metrics for the company
Cypher: MATCH (c:Company)-[:HAS_METRIC]->(m:Metric) RETURN m.name AS metric_name, m.value AS metric_value

Question: What metrics does Zenalyst Inc. have?
Cypher: MATCH (c:Company {{name: 'Zenalyst Inc.'}})-[:HAS_METRIC]->(m:Metric) RETURN m.name, m.value
"""),
    ("human", "{question}")
])

# Create the Graph QA Chain with custom prompt
graph_qa_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=neo4j_graph,
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_requests=True,
    cypher_prompt=cypher_generation_prompt
)

# --- 2. Vector Search (Unstructured Data Retrieval) ---

try:
    # This retriever searches the PDF chunks you loaded in Phase 1 (Node: Chunk)
    vector_retriever = Neo4jVector.from_existing_index(
        embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name="pdf_vector_index",
        node_label="Chunk",
        text_node_property="text",
        embedding_node_property="embedding",
        retrieval_query="""
            RETURN node.text AS text, score, {source: node.source_file, type: "PDF Chunk"} AS metadata
        """
    ).as_retriever(search_kwargs={"k": 3})
    vector_search_available = True
except Exception as e:
    st.warning(f"Vector search not available: {e}")
    vector_search_available = False


# --- 3. Hybrid RAG (The Intelligence Agent) ---

def run_hybrid_query(question):
    """
    Routes queries to either Graph Search (structured) or Vector Search (unstructured)
    based on keywords in the question.
    """
    # Keywords that indicate a structured data query
    structured_keywords = ["value", "metric", "ratio", "equity", "debt", "rate", "show me", "what is the"]
    
    # Check if question contains structured query keywords
    is_structured = any(keyword in question.lower() for keyword in structured_keywords)
    
    if is_structured:
        st.info("🔍 Agent Action: Executing **Graph Search (Cypher)** for Structured Data.")
        
        try:
            # Run the Graph Chain
            graph_result = graph_qa_chain.invoke({"query": question})
            cypher_query = graph_result['intermediate_steps'][0]['query']
            
            # Display the generated Cypher
            st.code(cypher_query, language='cypher')
            st.caption("Generated Cypher Query")
            
            return graph_result['result'], cypher_query
            
        except Exception as e:
            error_msg = f"Graph search failed: {str(e)}\n\nTrying vector search instead..."
            st.warning(error_msg)
            # Fallback to vector search
            is_structured = False
    
    if not is_structured:
        st.info("📚 Agent Action: Executing **Vector Search** for Unstructured Context.")
        
        if not vector_search_available:
            return "Vector search is not available. Please run scripts/pdf.py to create the vector index.", None
        
        try:
            # Run the Vector Retriever
            vector_docs = vector_retriever.invoke(question)
            
            if not vector_docs:
                return "No relevant information found in the PDF documents.", None
            
            # Combine retrieved context
            context = "\n\n".join([doc.page_content for doc in vector_docs])
            
            # Show retrieved chunks in expander
            with st.expander("📄 Retrieved Context"):
                for i, doc in enumerate(vector_docs, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(doc.page_content[:300] + "...")
            
            # Define the final RAG chain prompt
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a professional analyst reviewing government tax audit documentation. 
Use ONLY the following contextual information to answer the question. 
Do not make up information. If the context doesn't contain the answer, say so.

CONTEXT:
{context}"""),
                ("human", "Question: {question}")
            ])

            # Combine components using LangChain Expression Language (LCEL)
            rag_chain = (
                RunnableParallel(
                    context=lambda x: context,
                    question=RunnablePassthrough()
                ) | rag_prompt | llm | StrOutputParser()
            )
            
            vector_answer = rag_chain.invoke({"question": question})
            return vector_answer, None
            
        except Exception as e:
            return f"Vector search failed: {str(e)}", None

# --- STREAMLIT UI ---
st.set_page_config(page_title="Advanced Graph RAG Demo", layout="wide")
st.title("🧠 Advanced Graph RAG Demo (Gemini + Neo4j)")
st.caption("A Hybrid AI Agent combining Structured Data (Excel) and Unstructured Context (PDF)")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This demo showcases hybrid RAG:
    - **Graph Search**: For structured metrics (Excel data)
    - **Vector Search**: For unstructured context (PDF content)
    
    **Example Questions:**
    
    *Structured (Graph):*
    - What is the Equity value?
    - Show all metrics
    
    *Unstructured (Vector):*
    - What weaknesses were found?
    - What are the Collections recommendations?
    """)
    
    # Debug info
    if st.checkbox("Show Debug Info"):
        st.code(f"Neo4j URI: {NEO4J_URI}")
        st.code(f"LLM Model: {LLM_MODEL}")
        st.code(f"Vector Search: {'Available' if vector_search_available else 'Not Available'}")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "details" in message:
            st.code(message["details"], language='cypher')
            st.caption("Cypher Query Used")

# Chat input
if prompt := st.chat_input("Ask a question about the company's financial data or annual report..."):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query
    with st.spinner("🤔 Thinking..."):
        try:
            final_answer, cypher_query = run_hybrid_query(prompt)
        except Exception as e:
            final_answer = f"Error processing query: {str(e)}"
            cypher_query = None

    # Store AI response
    ai_message = {"role": "assistant", "content": final_answer}
    if cypher_query:
        ai_message["details"] = cypher_query
    
    st.session_state.messages.append(ai_message)
    
    # Display the final response
    with st.chat_message("assistant"):
        st.markdown(final_answer)
        if cypher_query:
            st.code(cypher_query, language='cypher')
            st.caption("Cypher Query Used")