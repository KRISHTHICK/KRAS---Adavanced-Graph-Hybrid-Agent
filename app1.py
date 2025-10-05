# # # # import os
# # # # import streamlit as st
# # # # from dotenv import load_dotenv
# # # # from langchain_core.prompts import ChatPromptTemplate
# # # # from langchain_core.output_parsers import StrOutputParser
# # # # from langchain_core.runnables import RunnablePassthrough, RunnableParallel
# # # # from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# # # # from langchain_community.graphs import Neo4jGraph
# # # # from langchain_neo4j import Neo4jVector
# # # # from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

# # # # # Import the custom ingestion logic from your scripts folder
# # # # from scripts.ingest_utility import ingest_data, remove_data

# # # # # --- 0. Setup and Initialization ---
# # # # load_dotenv()

# # # # # Retrieve credentials and settings from .env
# # # # GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
# # # # NEO4J_URI = os.getenv("NEO4J_URI")
# # # # NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# # # # NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
# # # # LLM_MODEL = os.getenv("LLM_MODEL")
# # # # EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# # # # # --- Initialize Core Components (LLM, Embeddings, Neo4j) ---

# # # # # Validate credentials
# # # # if not all([GOOGLE_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
# # # #     st.error("Missing environment variables. Please check your .env file.")
# # # #     st.stop()

# # # # # Cache resources that do not change during runtime
# # # # @st.cache_resource
# # # # def initialize_rag_components():
# # # #     # Gemini Chat Model (for generation, Cypher translation, and Graph Structuring)
# # # #     llm_instance = ChatGoogleGenerativeAI(model=LLM_MODEL, api_key=GOOGLE_KEY, temperature=0)

# # # #     # Gemini Embeddings Model (for vector search)
# # # #     embeddings_instance = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=GOOGLE_KEY)

# # # #     # Neo4j Graph Connection
# # # #     neo4j_graph_instance = Neo4jGraph(
# # # #         url=NEO4J_URI,
# # # #         username=NEO4J_USERNAME,
# # # #         password=NEO4J_PASSWORD
# # # #     )
    
# # # #     # Ensure the schema is refreshed for the LLM's query generation tool
# # # #     neo4j_graph_instance.refresh_schema()

# # # #     return llm_instance, embeddings_instance, neo4j_graph_instance

# # # # llm, embeddings, neo4j_graph = initialize_rag_components()

# # # # # --- 1. Graph Search (Structured Data Retrieval) ---

# # # # # Custom Cypher prompt with better examples for financial data
# # # # cypher_generation_prompt = ChatPromptTemplate.from_messages([
# # # #     ("system", """Task: Generate Cypher statement to query a graph database.
# # # # Instructions:
# # # # - Use only the provided relationship types and properties in the schema
# # # # - Do not use any other relationship types or properties that are not provided
# # # # - Write only ONE Cypher query with ONE RETURN clause at the end
# # # # - Be precise with property names and node labels

# # # # Schema:
# # # # {schema}

# # # # Examples:
# # # # Question: What is the Equity value for Zenalyst Corp.?
# # # # Cypher: MATCH (c:Company {name: 'Zenalyst Corp.'})-[:HAS_METRIC]->(m:Metric {name: 'Equity'}) RETURN m.value AS equity_value

# # # # Question: Show all metrics for the company
# # # # Cypher: MATCH (c:Company)-[:HAS_METRIC]->(m:Metric) RETURN m.name AS metric_name, m.value AS metric_value

# # # # Question: What concepts are discussed?
# # # # Cypher: MATCH (c:Concept) RETURN c.name AS concept_name

# # # # """),
# # # #     ("human", "{question}")
# # # # ])

# # # # # Create the Graph QA Chain with custom prompt
# # # # graph_qa_chain = GraphCypherQAChain.from_llm(
# # # #     llm=llm,
# # # #     graph=neo4j_graph,
# # # #     verbose=True,
# # # #     return_intermediate_steps=True,
# # # #     allow_dangerous_requests=True,
# # # #     cypher_prompt=cypher_generation_prompt
# # # # )

# # # # # --- 2. Vector Search (Unstructured Data Retrieval) ---

# # # # @st.cache_resource(ttl=3600)
# # # # def setup_vector_retriever():
# # # #     """Initializes and returns the vector retriever for PDF chunks."""
# # # #     try:
# # # #         # This retriever searches the PDF chunks (Node: Chunk)
# # # #         vector_retriever = Neo4jVector.from_existing_index(
# # # #             embeddings,
# # # #             url=NEO4J_URI,
# # # #             username=NEO4J_USERNAME,
# # # #             password=NEO4J_PASSWORD,
# # # #             index_name="pdf_vector_index",
# # # #             node_label="Chunk",
# # # #             text_node_property="text",
# # # #             embedding_node_property="embedding",
# # # #             retrieval_query="""
# # # #                  # This query returns the text and the file source metadata
# # # #                  RETURN node.text AS text, score, {source: node.source_file, type: "PDF Chunk"} AS metadata
# # # #              """
# # # #         ).as_retriever(search_kwargs={"k": 3})
# # # #         return vector_retriever, True
# # # #     except Exception as e:
# # # #         # This occurs if the Vector Index has not been created yet (i.e., no PDF uploaded)
# # # #         return str(e), False

# # # # vector_retriever, vector_search_available = setup_vector_retriever()

# # # # # --- 3. Hybrid RAG (The Intelligence Agent) ---

# # # # def run_hybrid_query(question):
# # # #     """
# # # #     Routes queries to either Graph Search (structured/relational) 
# # # #     or Vector Search (unstructured/semantic) based on keywords.
# # # #     """
# # # #     # Keywords that indicate a structured or relational query
# # # #     structured_keywords = ["value", "metric", "ratio", "equity", "debt", "rate", "company", "who", "what is the"]
    
# # # #     is_structured = any(keyword in question.lower() for keyword in structured_keywords)
    
# # # #     if is_structured:
# # # #         st.info("🔍 Agent Action: Executing **Graph Search (Cypher)** for Structured/Relational Data.")
        
# # # #         try:
# # # #             # Run the Graph Chain
# # # #             graph_result = graph_qa_chain.invoke({"query": question})
# # # #             cypher_query = graph_result['intermediate_steps'][0]['query']
            
# # # #             # Display the generated Cypher in the chat panel
# # # #             st.code(cypher_query, language='cypher')
# # # #             st.caption("Generated Cypher Query")
            
# # # #             return graph_result['result'], cypher_query
            
# # # #         except Exception as e:
# # # #             error_msg = f"Graph search failed (Cypher error or hallucination): {str(e)}\n\nAttempting Vector Search fallback."
# # # #             st.warning(error_msg)
# # # #             # Fallback to vector search if graph query fails (e.g., bad Cypher generated)
# # # #             is_structured = False
    
# # # #     if not is_structured:
# # # #         st.info("📚 Agent Action: Executing **Vector Search** for Unstructured Context.")
        
# # # #         if not vector_search_available:
# # # #             return "Vector search is not available. Please upload a PDF to create the vector index.", None
        
# # # #         try:
# # # #             # Run the Vector Retriever
# # # #             vector_docs = vector_retriever.invoke(question)
            
# # # #             if not vector_docs or not vector_docs[0].page_content.strip():
# # # #                 return "No relevant contextual information found in the documents.", None
            
# # # #             # Combine retrieved context
# # # #             context = "\n\n".join([doc.page_content for doc in vector_docs])
            
# # # #             # Show retrieved chunks in expander
# # # #             with st.expander("📄 Retrieved Context"):
# # # #                 for i, doc in enumerate(vector_docs, 1):
# # # #                     st.markdown(f"**Chunk {i}:**")
# # # #                     st.text(doc.page_content[:300] + "...")
            
# # # #             # Define the final RAG chain prompt
# # # #             rag_prompt = ChatPromptTemplate.from_messages([
# # # #                 ("system", """You are a professional financial analyst. Your goal is to answer the user's question accurately using ONLY the CONTEXT provided. Do not invent information. If the context does not contain the answer, state that you cannot find the relevant information.
# # # # CONTEXT:
# # # # {context}"""),
# # # #                 ("human", "Question: {question}")
# # # #             ])

# # # #             # Combine components using LangChain Expression Language (LCEL)
# # # #             rag_chain = (
# # # #                 RunnableParallel(
# # # #                     context=lambda x: context,
# # # #                     question=RunnablePassthrough()
# # # #                 ) | rag_prompt | llm | StrOutputParser()
# # # #             )
            
# # # #             vector_answer = rag_chain.invoke({"question": question})
# # # #             return vector_answer, None
            
# # # #         except Exception as e:
# # # #             return f"Vector search failed: {str(e)}", None

# # # # # --- STREAMLIT UI ---
# # # # st.set_page_config(page_title="Advanced Graph RAG Demo", layout="wide")
# # # # st.title("🧠 Advanced Graph RAG Demo (Gemini + Neo4j)")
# # # # st.caption("A Hybrid AI Agent combining Structured Data (Excel metrics) and Unstructured Context (PDF chunks)")

# # # # # --- Sidebar for File Management ---
# # # # with st.sidebar:
# # # #     st.header("Upload/Remove Documents")
    
# # # #     # 1. Upload Logic
# # # #     uploaded_file = st.file_uploader("Upload Excel (.xlsx) or PDF (.pdf)", type=['xlsx', 'pdf'])
# # # #     company_input = st.text_input("Company Name for Tagging:", value="Zenalyst Corp.")
    
# # # #     if uploaded_file and st.button("🚀 Ingest & Process Data"):
# # # #         file_ext = uploaded_file.name.split('.')[-1]
# # # #         if file_ext in ['xlsx', 'pdf']:
# # # #             with st.spinner(f"Ingesting {uploaded_file.name} (This creates vectors and graph entities)..."):
# # # #                 # Call the robust ingestion function
# # # #                 ingestion_status = ingest_data(uploaded_file, file_ext, company_input)
# # # #                 st.session_state['messages'] = [] # Clear chat history after ingestion
# # # #                 st.cache_resource.clear() # Clear LangChain cache to pick up new index
# # # #                 st.success(ingestion_status)
# # # #                 st.rerun() # Rerun to update the retriever status
# # # #         else:
# # # #             st.error("Please upload a valid .xlsx or .pdf file.")

# # # #     st.markdown("---")
    
# # # #     # 2. Remove Logic
# # # #     file_to_remove = st.text_input("File Name to Remove:", help="Enter the exact file name (e.g., Annual_Report.pdf)")
# # # #     if st.button("🗑️ Remove Data"):
# # # #         if file_to_remove:
# # # #             with st.spinner(f"Removing data for {file_to_remove}..."):
# # # #                 removal_status = remove_data(file_to_remove)
# # # #                 st.session_state['messages'] = [] # Clear chat history after removal
# # # #                 st.cache_resource.clear()
# # # #                 st.success(removal_status)
# # # #                 st.rerun()
# # # #         else:
# # # #             st.warning("Please enter a file name to remove.")
            
# # # #     st.header("ℹ️ Example Queries")
# # # #     st.markdown("""
# # # #     *Structured (Graph, Excel/PDF entities):*
# # # #     - What is the Equity value?
# # # #     - Show all metrics for Zenalyst Corp.
# # # #     - What are the risks mentioned?
    
# # # #     *Unstructured (Vector, PDF content):*
# # # #     - What weaknesses were found in the audit?
# # # #     - Describe the Collections recommendations.
# # # #     """)

# # # # # --- Chat Interface ---
# # # # if "messages" not in st.session_state:
# # # #     st.session_state.messages = []

# # # # # Display chat history
# # # # for message in st.session_state.messages:
# # # #     with st.chat_message(message["role"]):
# # # #         st.markdown(message["content"])
# # # #         if "details" in message:
# # # #             st.code(message["details"], language='cypher')
# # # #             st.caption("Cypher Query Used")

# # # # # Chat input
# # # # if prompt := st.chat_input("Ask a question about the documents..."):
# # # #     # Store user message
# # # #     st.session_state.messages.append({"role": "user", "content": prompt})
# # # #     with st.chat_message("user"):
# # # #         st.markdown(prompt)

# # # #     # Process query
# # # #     with st.spinner("🤔 Thinking..."):
# # # #         try:
# # # #             final_answer, cypher_query = run_hybrid_query(prompt)
# # # #         except Exception as e:
# # # #             final_answer = f"Error processing query: {str(e)}"
# # # #             cypher_query = None

# # # #     # Store AI response
# # # #     ai_message = {"role": "assistant", "content": final_answer}
# # # #     if cypher_query:
# # # #         ai_message["details"] = cypher_query
    
# # # #     st.session_state.messages.append(ai_message)
    
# # # #     # Display the final response
# # # #     with st.chat_message("assistant"):
# # # #         st.markdown(final_answer)
# # # #         if cypher_query:
# # # #             st.code(cypher_query, language='cypher')
# # # #             st.caption("Cypher Query Used")


# # # import os
# # # import streamlit as st
# # # from dotenv import load_dotenv
# # # from langchain_core.prompts import ChatPromptTemplate
# # # from langchain_core.output_parsers import StrOutputParser
# # # from langchain_core.runnables import RunnablePassthrough, RunnableParallel
# # # from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# # # from langchain_community.graphs import Neo4jGraph
# # # from langchain_neo4j import Neo4jVector
# # # from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

# # # from scripts.ingest_utility import ingest_data, remove_data

# # # # --- 0. Setup and Initialization ---
# # # load_dotenv()

# # # GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
# # # NEO4J_URI = os.getenv("NEO4J_URI")
# # # NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# # # NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
# # # LLM_MODEL = os.getenv("LLM_MODEL")
# # # EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# # # if not all([GOOGLE_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
# # #     st.error("Missing environment variables. Please check your .env file.")
# # #     st.stop()

# # # @st.cache_resource
# # # def initialize_rag_components():
# # #     llm_instance = ChatGoogleGenerativeAI(model=LLM_MODEL, api_key=GOOGLE_KEY, temperature=0)
# # #     embeddings_instance = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=GOOGLE_KEY)
# # #     neo4j_graph_instance = Neo4jGraph(
# # #         url=NEO4J_URI,
# # #         username=NEO4J_USERNAME,
# # #         password=NEO4J_PASSWORD
# # #     )
# # #     neo4j_graph_instance.refresh_schema()
# # #     return llm_instance, embeddings_instance, neo4j_graph_instance

# # # llm, embeddings, neo4j_graph = initialize_rag_components()

# # # # --- 1. Graph Search (Structured Data Retrieval) ---

# # # cypher_generation_prompt = ChatPromptTemplate.from_messages([
# # #     ("system", """Task: Generate Cypher statement to query a graph database.
# # # Instructions:
# # # - Use only the provided relationship types and properties in the schema
# # # - Write only ONE Cypher query with ONE RETURN clause at the end
# # # - Be precise with property names and node labels

# # # Schema:
# # # {schema}

# # # Examples:
# # # Question: What is the Equity value?
# # # Cypher: MATCH (c:Company)-[:HAS_METRIC]->(m:Metric {{name: 'Equity'}}) RETURN m.value AS equity_value

# # # Question: Show all metrics for the company
# # # Cypher: MATCH (c:Company)-[:HAS_METRIC]->(m:Metric) RETURN m.name AS metric_name, m.value AS metric_value

# # # Question: What are the total assets?
# # # Cypher: MATCH (c:Company)-[:HAS_METRIC]->(m:Metric {{name: 'Total Assets'}}) RETURN m.value AS total_assets
# # # """),
# # #     ("human", "{question}")
# # # ])

# # # graph_qa_chain = GraphCypherQAChain.from_llm(
# # #     llm=llm,
# # #     graph=neo4j_graph,
# # #     verbose=True,
# # #     return_intermediate_steps=True,
# # #     allow_dangerous_requests=True,
# # #     cypher_prompt=cypher_generation_prompt
# # # )

# # # # --- 2. Vector Search (Unstructured Data Retrieval) ---

# # # @st.cache_resource(ttl=3600)
# # # def setup_vector_retriever():
# # #     try:
# # #         vector_retriever = Neo4jVector.from_existing_index(
# # #             embeddings,
# # #             url=NEO4J_URI,
# # #             username=NEO4J_USERNAME,
# # #             password=NEO4J_PASSWORD,
# # #             index_name="pdf_vector_index",
# # #             node_label="Chunk",
# # #             text_node_property="text",
# # #             embedding_node_property="embedding",
# # #             retrieval_query="""
# # #                 RETURN node.text AS text, score, {source: node.source_file, type: "PDF Chunk"} AS metadata
# # #             """
# # #         ).as_retriever(search_kwargs={"k": 3})
# # #         return vector_retriever, True
# # #     except Exception as e:
# # #         return str(e), False

# # # vector_retriever, vector_search_available = setup_vector_retriever()

# # # # --- 3. LLM-Based Query Router ---

# # # def intelligent_query_router(question):
# # #     """Uses LLM to decide between graph (structured) or vector (unstructured) search."""
# # #     router_prompt = ChatPromptTemplate.from_messages([
# # #         ("system", """You are a query routing assistant. Your job is to classify user questions into TWO categories:

# # # 1. GRAPH - Questions asking for specific data points, numbers, metrics, comparisons, or relationships between entities
# # #    Examples: "What is the equity value?", "Show all metrics", "Compare debt and equity", "List all companies"

# # # 2. VECTOR - Questions asking for explanations, summaries, recommendations, or contextual information from documents
# # #    Examples: "What weaknesses were found?", "Summarize the report", "What are the recommendations?", "Explain the findings"

# # # Respond with ONLY one word: either "GRAPH" or "VECTOR"
# # # """),
# # #         ("human", "Question: {question}\n\nClassification:")
# # #     ])
    
# # #     router_chain = router_prompt | llm | StrOutputParser()
# # #     decision = router_chain.invoke({"question": question}).strip().upper()
    
# # #     return "GRAPH" if "GRAPH" in decision else "VECTOR"

# # # # --- 4. Hybrid RAG (The Intelligence Agent) ---

# # # def run_hybrid_query(question):
# # #     """Routes queries using LLM-based intelligent routing."""
    
# # #     # Use LLM to decide routing
# # #     route_decision = intelligent_query_router(question)
    
# # #     if route_decision == "GRAPH":
# # #         st.info("🔍 Agent Action: Executing **Graph Search (Cypher)** for Structured/Relational Data.")
        
# # #         try:
# # #             graph_result = graph_qa_chain.invoke({"query": question})
# # #             cypher_query = graph_result['intermediate_steps'][0]['query']
            
# # #             st.code(cypher_query, language='cypher')
# # #             st.caption("Generated Cypher Query")
            
# # #             return graph_result['result'], cypher_query
            
# # #         except Exception as e:
# # #             error_msg = f"Graph search failed: {str(e)}\n\nFalling back to Vector Search..."
# # #             st.warning(error_msg)
# # #             route_decision = "VECTOR"
    
# # #     if route_decision == "VECTOR":
# # #         st.info("📚 Agent Action: Executing **Vector Search** for Unstructured Context.")
        
# # #         if not vector_search_available:
# # #             return "Vector search is not available. Please upload a PDF to create the vector index.", None
        
# # #         try:
# # #             vector_docs = vector_retriever.invoke(question)
            
# # #             if not vector_docs or not vector_docs[0].page_content.strip():
# # #                 return "No relevant information found in the documents.", None
            
# # #             context = "\n\n".join([doc.page_content for doc in vector_docs])
            
# # #             with st.expander("📄 Retrieved Context"):
# # #                 for i, doc in enumerate(vector_docs, 1):
# # #                     st.markdown(f"**Chunk {i}:**")
# # #                     st.text(doc.page_content[:300] + "...")
            
# # #             rag_prompt = ChatPromptTemplate.from_messages([
# # #                 ("system", """You are a professional analyst. Use ONLY the following context to answer the question accurately.
# # # Do not invent information. If the context doesn't contain the answer, state that clearly.

# # # CONTEXT:
# # # {context}"""),
# # #                 ("human", "Question: {question}")
# # #             ])

# # #             rag_chain = (
# # #                 RunnableParallel(
# # #                     context=lambda x: context,
# # #                     question=RunnablePassthrough()
# # #                 ) | rag_prompt | llm | StrOutputParser()
# # #             )
            
# # #             vector_answer = rag_chain.invoke({"question": question})
# # #             return vector_answer, None
            
# # #         except Exception as e:
# # #             return f"Vector search failed: {str(e)}", None

# # # # --- STREAMLIT UI ---
# # # st.set_page_config(page_title="Advanced Graph RAG Demo", layout="wide")
# # # st.title("🧠 Advanced Graph RAG Demo (Gemini + Neo4j)")
# # # st.caption("A Hybrid AI Agent combining Structured Data (Excel metrics) and Unstructured Context (PDF chunks)")

# # # # --- Sidebar for File Management ---
# # # with st.sidebar:
# # #     st.header("Upload/Remove Documents")
    
# # #     uploaded_file = st.file_uploader("Upload Excel (.xlsx) or PDF (.pdf)", type=['xlsx', 'pdf'])
# # #     company_input = st.text_input("Company Name for Tagging:", value="ABC Book Stores")
    
# # #     if uploaded_file and st.button("🚀 Ingest & Process Data"):
# # #         file_ext = uploaded_file.name.split('.')[-1]
# # #         if file_ext in ['xlsx', 'pdf']:
# # #             with st.spinner(f"Ingesting {uploaded_file.name}..."):
# # #                 ingestion_status = ingest_data(uploaded_file, file_ext, company_input)
# # #                 st.session_state['messages'] = []
# # #                 st.cache_resource.clear()
# # #                 st.success(ingestion_status)
# # #                 st.rerun()
# # #         else:
# # #             st.error("Please upload a valid .xlsx or .pdf file.")

# # #     st.markdown("---")
    
# # #     file_to_remove = st.text_input("File Name to Remove:", help="Enter exact file name")
# # #     if st.button("🗑️ Remove Data"):
# # #         if file_to_remove:
# # #             with st.spinner(f"Removing {file_to_remove}..."):
# # #                 removal_status = remove_data(file_to_remove)
# # #                 st.session_state['messages'] = []
# # #                 st.cache_resource.clear()
# # #                 st.success(removal_status)
# # #                 st.rerun()
# # #         else:
# # #             st.warning("Please enter a file name.")
            
# # #     st.header("ℹ️ Example Queries")
# # #     st.markdown("""
# # # **Structured (Graph):**
# # # - What is the total opening stock amount?
# # # - Show all book titles
# # # - List all invoices
# # # - What is the GST for invoice INV-202510-017?

# # # **Unstructured (Vector):**
# # # - Summarize invoices from June 2024
# # # - What books are in Jayanagar store?
# # # - List all authors in inventory
# # # - Describe the purchase orders
# # # """)

# # # # --- Chat Interface ---
# # # if "messages" not in st.session_state:
# # #     st.session_state.messages = []

# # # for message in st.session_state.messages:
# # #     with st.chat_message(message["role"]):
# # #         st.markdown(message["content"])
# # #         if "details" in message:
# # #             st.code(message["details"], language='cypher')
# # #             st.caption("Cypher Query Used")

# # # if prompt := st.chat_input("Ask a question about your data..."):
# # #     st.session_state.messages.append({"role": "user", "content": prompt})
# # #     with st.chat_message("user"):
# # #         st.markdown(prompt)

# # #     with st.spinner("🤔 Thinking..."):
# # #         try:
# # #             final_answer, cypher_query = run_hybrid_query(prompt)
# # #         except Exception as e:
# # #             final_answer = f"Error: {str(e)}"
# # #             cypher_query = None

# # #     ai_message = {"role": "assistant", "content": final_answer}
# # #     if cypher_query:
# # #         ai_message["details"] = cypher_query
    
# # #     st.session_state.messages.append(ai_message)
    
# # #     with st.chat_message("assistant"):
# # #         st.markdown(final_answer)
# # #         if cypher_query:
# # #             st.code(cypher_query, language='cypher')
# # #             st.caption("Cypher Query Used")

# # # # Plan Final 
# # # import os
# # # import streamlit as st
# # # from dotenv import load_dotenv
# # # from langchain_core.prompts import ChatPromptTemplate
# # # from langchain_core.output_parsers import StrOutputParser
# # # from langchain_core.runnables import RunnablePassthrough, RunnableParallel
# # # from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# # # from langchain_community.graphs import Neo4jGraph
# # # from langchain_neo4j import Neo4jVector
# # # from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

# # # # Import the custom ingestion logic from your scripts folder
# # # # NOTE: The helper functions get_llm, get_embeddings, etc., are imported here but defined
# # # # to be called lazily in the utility file to ensure environment variables are loaded first.
# # # from scripts.ingest_utility import ingest_data, remove_data, get_llm, get_embeddings, get_neo4j_graph

# # # # --- 0. Setup and Initialization ---
# # # load_dotenv()

# # # # Retrieve credentials and settings from .env
# # # GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
# # # NEO4J_URI = os.getenv("NEO4J_URI")
# # # NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# # # NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
# # # LLM_MODEL = os.getenv("LLM_MODEL")
# # # EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# # # # Validate credentials
# # # if not all([GOOGLE_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
# # #     st.error("Missing environment variables. Please check your .env file.")
# # #     st.stop()

# # # # Cache resources that do not change during runtime
# # # @st.cache_resource
# # # def initialize_rag_components():
# # #     # We use the utility functions to get lazily initialized components
# # #     llm_instance = get_llm()
# # #     embeddings_instance = get_embeddings()
# # #     neo4j_graph_instance = get_neo4j_graph()
    
# # #     # Ensure the schema is refreshed for the LLM's query generation tool
# # #     neo4j_graph_instance.refresh_schema()

# # #     return llm_instance, embeddings_instance, neo4j_graph_instance

# # # llm, embeddings, neo4j_graph = initialize_rag_components()

# # # # --- 1. Graph Search (Structured Data Retrieval) ---

# # # # UPDATED: Cypher prompt includes examples for the new Book/Metric/Store structure
# # # cypher_generation_prompt = ChatPromptTemplate.from_messages([
# # #     ("system", """Task: Generate Cypher statement to query a graph database.
# # # Instructions:
# # # - Use only the provided relationship types, node labels (e.g., Company, Metric, Book, Store), and properties in the schema.
# # # - Write only ONE Cypher query with ONE RETURN clause at the end.
# # # - Be precise with property names and node labels.

# # # Schema:
# # # {schema}

# # # Examples:
# # # Question: What is the total purchase amount?
# # # Cypher: MATCH (m:Metric {name: 'Total Purchase Amount'}) RETURN m.value AS total_purchase

# # # Question: List all book titles
# # # Cypher: MATCH (b:Book) RETURN b.title AS book_title

# # # Question: What books are in the Jayanagar store?
# # # Cypher: MATCH (s:Store {name: 'Bangalore – Jayanagar'})<-[:LOCATED_AT]-(b:Book) RETURN b.title AS book_title

# # # Question: What is the total opening stock amount?
# # # Cypher: MATCH (m:Metric {name: 'Total Opening Stock Amount'}) RETURN m.value AS total_opening_stock
# # # """),
# # #     ("human", "{question}")
# # # ])

# # # graph_qa_chain = GraphCypherQAChain.from_llm(
# # #     llm=llm,
# # #     graph=neo4j_graph,
# # #     verbose=True,
# # #     return_intermediate_steps=True,
# # #     allow_dangerous_requests=True,
# # #     cypher_prompt=cypher_generation_prompt
# # # )

# # # # --- 2. Vector Search (Unstructured Data Retrieval) ---

# # # @st.cache_resource(ttl=3600)
# # # def setup_vector_retriever():
# # #     """Initializes and returns the vector retriever for PDF chunks."""
# # #     try:
# # #         # This retriever searches the PDF chunks (Node: Chunk)
# # #         vector_retriever = Neo4jVector.from_existing_index(
# # #             embeddings,
# # #             url=NEO4J_URI,
# # #             username=NEO4J_USERNAME,
# # #             password=NEO4J_PASSWORD,
# # #             index_name="pdf_vector_index",
# # #             node_label="Chunk",
# # #             text_node_property="text",
# # #             embedding_node_property="embedding",
# # #             retrieval_query="""
# # #                  RETURN node.text AS text, score, {source: node.source_file, type: "PDF Chunk"} AS metadata
# # #              """
# # #         ).as_retriever(search_kwargs={"k": 3})
# # #         return vector_retriever, True
# # #     except Exception as e:
# # #         return str(e), False

# # # vector_retriever, vector_search_available = setup_vector_retriever()


# # # # --- 3. LLM-Based Query Router (Intelligent Routing) ---

# # # def intelligent_query_router(question):
# # #     """Uses LLM to decide between graph (structured) or vector (unstructured) search."""
# # #     # We use a cleaner version of the prompt that relies on the LLM's understanding
# # #     # of the core question intent (numerical vs. contextual).
# # #     router_prompt = ChatPromptTemplate.from_messages([
# # #         ("system", """You are a precise query routing assistant. Classify the user's question into ONE of TWO categories:

# # # 1. GRAPH: Questions asking for specific numbers, totals, amounts, lists of inventory, comparisons, or entity relationships (e.g., "What is the total amount?", "List all books", "Which author is linked to which category?").
# # # 2. VECTOR: Questions asking for summaries, descriptions, explanations, or contextual narrative (e.g., "Summarize the orders from June", "What are the issues?", "Explain the purchase orders").

# # # Respond with ONLY one word: "GRAPH" or "VECTOR".
# # # """),
# # #         ("human", "Question: {question}\n\nClassification:")
# # #     ])
    
# # #     router_chain = router_prompt | llm | StrOutputParser()
# # #     decision = router_chain.invoke({"question": question}).strip().upper()
    
# # #     # We check if 'GRAPH' is in the decision to be robust against any extra words from the LLM
# # #     return "GRAPH" if "GRAPH" in decision else "VECTOR"

# # # # --- 4. Hybrid RAG (The Intelligence Agent) ---

# # # def run_hybrid_query(question):
# # #     """Routes queries using LLM-based intelligent routing."""
    
# # #     route_decision = intelligent_query_router(question)
# # #     cypher_query = None
    
# # #     if route_decision == "GRAPH":
# # #         st.info("🔍 Agent Action: Executing **Graph Search (Cypher)** for Structured/Relational Data.")
        
# # #         try:
# # #             # 1. Run the Graph Chain (NL -> Cypher -> Query -> LLM Answer)
# # #             graph_result = graph_qa_chain.invoke({"query": question})
# # #             cypher_query = graph_result['intermediate_steps'][0]['query']
            
# # #             # The result is the final NL answer generated by the LLM
# # #             return graph_result['result'], cypher_query
            
# # #         except Exception as e:
# # #             error_msg = f"Graph search failed (Cypher error or hallucination): {str(e)}\n\nFalling back to Vector Search..."
# # #             st.warning(error_msg)
# # #             route_decision = "VECTOR"
# # #             # Proceed to VECTOR path if graph fails

# # #     if route_decision == "VECTOR":
# # #         st.info("📚 Agent Action: Executing **Vector Search** for Unstructured Context.")
        
# # #         if not vector_search_available:
# # #             return "Vector search is not available. Please upload a PDF to create the vector index.", None
        
# # #         try:
# # #             # 1. Retrieve Context
# # #             vector_docs = vector_retriever.invoke(question)
            
# # #             if not vector_docs or not vector_docs[0].page_content.strip():
# # #                 return "No relevant contextual information found in the documents.", None
            
# # #             context = "\n\n".join([doc.page_content for doc in vector_docs])
            
# # #             # 2. Show retrieved chunks in expander
# # #             with st.expander("📄 Retrieved Context"):
# # #                 for i, doc in enumerate(vector_docs, 1):
# # #                     st.markdown(f"**Chunk {i}:**")
# # #                     st.text(doc.page_content[:300] + "...")
            
# # #             # 3. Final Answer Generation (RAG Chain)
# # #             rag_prompt = ChatPromptTemplate.from_messages([
# # #                 ("system", """You are a professional analyst. Use ONLY the following context to answer the question accurately.
# # # Do not invent information. If the context doesn't contain the answer, state that clearly.

# # # CONTEXT:
# # # {context}"""),
# # #                 ("human", "Question: {question}")
# # #             ])

# # #             rag_chain = (
# # #                 RunnableParallel(
# # #                     context=lambda x: context,
# # #                     question=RunnablePassthrough()
# # #                 ) | rag_prompt | llm | StrOutputParser()
# # #             )
            
# # #             vector_answer = rag_chain.invoke({"question": question})
# # #             return vector_answer, None
            
# # #         except Exception as e:
# # #             return f"Vector search failed: {str(e)}", None

# # # # --- STREAMLIT UI ---
# # # st.set_page_config(page_title="Advanced Graph RAG Demo", layout="wide")
# # # st.title("🧠 Advanced Graph RAG Demo (Gemini + Neo4j)")
# # # st.caption("A Hybrid AI Agent for Finance/Inventory Analysis")

# # # # --- Sidebar for File Management ---
# # # with st.sidebar:
# # #     st.header("Upload/Remove Documents")
    
# # #     uploaded_file = st.file_uploader("Upload Excel (.xlsx) or PDF (.pdf)", type=['xlsx', 'pdf'])
# # #     company_input = st.text_input("Company Name for Tagging:", value="ABC Book Stores")
    
# # #     if uploaded_file and st.button("🚀 Ingest & Process Data"):
# # #         file_ext = uploaded_file.name.split('.')[-1]
# # #         if file_ext in ['xlsx', 'pdf']:
# # #             with st.spinner(f"Ingesting {uploaded_file.name} (This creates vectors and graph entities)..."):
# # #                 # Call the robust ingestion function
# # #                 ingestion_status = ingest_data(uploaded_file, file_ext, company_input)
# # #                 st.session_state['messages'] = []
# # #                 st.cache_resource.clear() 
# # #                 st.success(ingestion_status)
# # #                 st.rerun() 
# # #         else:
# # #             st.error("Please upload a valid .xlsx or .pdf file.")

# # #     st.markdown("---")
    
# # #     file_to_remove = st.text_input("File Name to Remove:", help="Enter exact file name")
# # #     if st.button("🗑️ Remove Data"):
# # #         if file_to_remove:
# # #             with st.spinner(f"Removing {file_to_remove}..."):
# # #                 removal_status = remove_data(file_to_remove)
# # #                 st.session_state['messages'] = []
# # #                 st.cache_resource.clear()
# # #                 st.success(removal_status)
# # #                 st.rerun()
# # #         else:
# # #             st.warning("Please enter a file name.")
            
# # #     st.header("ℹ️ Example Queries")
# # #     st.markdown("""
# # # **Structured (Graph):**
# # # - What is the total purchase amount?
# # # - Show all book titles
# # # - What books are in the Jayanagar store?
# # # - List all authors in inventory

# # # **Unstructured (Vector):**
# # # - Summarize the purchase orders
# # # - What issues were found in the audit?
# # # - Explain the key findings.
# # # """)

# # # # --- Chat Interface ---
# # # if "messages" not in st.session_state:
# # #     st.session_state.messages = []

# # # for message in st.session_state.messages:
# # #     with st.chat_message(message["role"]):
# # #         st.markdown(message["content"])
# # #         if "details" in message:
# # #             st.code(message["details"], language='cypher')
# # #             st.caption("Cypher Query Used")

# # # if prompt := st.chat_input("Ask a question about your data..."):
# # #     st.session_state.messages.append({"role": "user", "content": prompt})
# # #     with st.chat_message("user"):
# # #         st.markdown(prompt)

# # #     with st.spinner("🤔 Thinking..."):
# # #         try:
# # #             final_answer, cypher_query = run_hybrid_query(prompt)
# # #         except Exception as e:
# # #             final_answer = f"Error: {str(e)}"
# # #             cypher_query = None

# # #     ai_message = {"role": "assistant", "content": final_answer}
# # #     if cypher_query:
# # #         ai_message["details"] = cypher_query
    
# # #     st.session_state.messages.append(ai_message)
    
# # #     with st.chat_message("assistant"):
# # #         st.markdown(final_answer)
# # #         if cypher_query:
# # #             st.code(cypher_query, language='cypher')
# # #             st.caption("Cypher Query Used")


# # # final plus code
# # import os
# # import streamlit as st
# # from dotenv import load_dotenv
# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain_core.output_parsers import StrOutputParser
# # from langchain_core.runnables import RunnablePassthrough, RunnableParallel
# # from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# # from langchain_community.graphs import Neo4jGraph
# # from langchain_neo4j import Neo4jVector
# # from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

# # # Import the custom ingestion logic from your scripts folder
# # # NOTE: The helper functions get_llm, get_embeddings, etc., are imported here but defined
# # # to be called lazily in the utility file to ensure environment variables are loaded first.
# # from scripts.ingest_utility import ingest_data, remove_data, get_llm, get_embeddings, get_neo4j_graph

# # # --- 0. Setup and Initialization ---
# # load_dotenv()

# # # Retrieve credentials and settings from .env
# # GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
# # NEO4J_URI = os.getenv("NEO4J_URI")
# # NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# # NEO4J_PASSWORD = os.getenv("NEO4AJ_PASSWORD") # Ensure this matches your utility file
# # LLM_MODEL = os.getenv("LLM_MODEL")
# # EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# # # Validate credentials
# # if not all([GOOGLE_KEY, NEO4J_URI, NEO4AJ_USERNAME, NEO4J_PASSWORD]):
# #     st.error("Missing environment variables. Please check your .env file.")
# #     st.stop()

# # # Cache resources that do not change during runtime
# # @st.cache_resource
# # def initialize_rag_components():
# #     # We use the utility functions to get lazily initialized components
# #     llm_instance = get_llm()
# #     embeddings_instance = get_embeddings()
# #     neo4j_graph_instance = get_neo4j_graph()
    
# #     # Ensure the schema is refreshed for the LLM's query generation tool
# #     neo4j_graph_instance.refresh_schema()

# #     return llm_instance, embeddings_instance, neo4j_graph_instance

# # llm, embeddings, neo4j_graph = initialize_rag_components()

# # # --- 1. Graph Search (Structured Data Retrieval) ---

# # # UPDATED: Cypher prompt includes examples for the new Book/Metric/Store structure
# # cypher_generation_prompt = ChatPromptTemplate.from_messages([
# #     ("system", """Task: Generate Cypher statement to query a graph database.
# # Instructions:
# # - Use only the provided relationship types, node labels (e.g., Company, Metric, Book, Store, Invoice), and properties in the schema.
# # - Write only ONE Cypher query with ONE RETURN clause at the end.
# # - Be precise with property names and node labels.

# # Schema:
# # {schema}

# # Examples:
# # Question: What is the total purchase amount?
# # Cypher: MATCH (m:Metric {name: 'Total Purchase Amount'}) RETURN m.value AS total_purchase

# # Question: List all book titles
# # Cypher: MATCH (b:Book) RETURN b.title AS book_title

# # Question: What books are in the Jayanagar store?
# # Cypher: MATCH (s:Store {name: 'Bangalore – Jayanagar'})<-[:LOCATED_AT]-(b:Book) RETURN b.title AS book_title

# # Question: What is the total opening stock amount?
# # Cypher: MATCH (m:Metric {name: 'Total Opening Stock Amount'}) RETURN m.value AS total_opening_stock
# # """),
# #     ("human", "{question}")
# # ])

# # graph_qa_chain = GraphCypherQAChain.from_llm(
# #     llm=llm,
# #     graph=neo4j_graph,
# #     verbose=True,
# #     return_intermediate_steps=True,
# #     allow_dangerous_requests=True,
# #     cypher_prompt=cypher_generation_prompt
# # )

# # # --- 2. Vector Search (Unstructured Data Retrieval) ---

# # @st.cache_resource(ttl=3600)
# # def setup_vector_retriever():
# #     """Initializes and returns the vector retriever for PDF chunks."""
# #     try:
# #         # This retriever searches the PDF chunks (Node: Chunk)
# #         vector_retriever = Neo4jVector.from_existing_index(
# #             embeddings,
# #             url=NEO4J_URI,
# #             username=NEO4J_USERNAME,
# #             password=NEO4J_PASSWORD,
# #             index_name="pdf_vector_index",
# #             node_label="Chunk",
# #             text_node_property="text",
# #             embedding_node_property="embedding",
# #             retrieval_query="""
# #                  RETURN node.text AS text, score, 
# #                     {source: node.source_file, type: "PDF Chunk"} AS metadata, node.company AS company
# #              """
# #         ).as_retriever(search_kwargs={"k": 3})
# #         return vector_retriever, True
# #     except Exception as e:
# #         return str(e), False

# # vector_retriever, vector_search_available = setup_vector_retriever()


# # # --- 3. LLM-Based Query Router (Intelligent Routing) ---

# # def intelligent_query_router(question):
# #     """Uses LLM to decide between graph (structured) or vector (unstructured) search."""
# #     # We use a cleaner version of the prompt that relies on the LLM's understanding
# #     # of the core question intent (numerical vs. contextual).
# #     router_prompt = ChatPromptTemplate.from_messages([
# #         ("system", """You are a precise query routing assistant. Classify the user's question into ONE of TWO categories:

# # 1. GRAPH: Questions asking for specific numbers, totals, amounts, lists of inventory, comparisons, or entity relationships (e.g., "What is the total amount?", "List all books", "Which author is linked to which category?").
# # 2. VECTOR: Questions asking for summaries, descriptions, explanations, or contextual narrative (e.g., "Summarize the orders from June", "What are the issues?", "Explain the purchase orders").

# # Respond with ONLY one word: "GRAPH" or "VECTOR".
# # """),
# #         ("human", "Question: {question}\n\nClassification:")
# #     ])
    
# #     router_chain = router_prompt | llm | StrOutputParser()
# #     decision = router_chain.invoke({"question": question}).strip().upper()
    
# #     # We check if 'GRAPH' is in the decision to be robust against any extra words from the LLM
# #     return "GRAPH" if "GRAPH" in decision else "VECTOR"

# # # --- 4. Hybrid RAG (The Intelligence Agent) ---

# # def run_hybrid_query(question):
# #     """Routes queries using LLM-based intelligent routing."""
    
# #     route_decision = intelligent_query_router(question)
# #     cypher_query = None
    
# #     if route_decision == "GRAPH":
# #         st.info("🔍 Agent Action: Executing **Graph Search (Cypher)** for Structured/Relational Data.")
        
# #         try:
# #             # 1. Run the Graph Chain (NL -> Cypher -> Query -> LLM Answer)
# #             graph_result = graph_qa_chain.invoke({"query": question})
# #             # LangChain returns intermediate steps as a list of dicts. We extract the generated Cypher query.
# #             cypher_query = graph_result['intermediate_steps'][0]['query']
            
# #             # The result is the final NL answer generated by the LLM
# #             return graph_result['result'], cypher_query
            
# #         except Exception as e:
# #             error_msg = f"Graph search failed (Cypher error or hallucination): {str(e)}\n\nFalling back to Vector Search..."
# #             st.warning(error_msg)
# #             route_decision = "VECTOR"
# #             # Proceed to VECTOR path if graph fails

# #     if route_decision == "VECTOR":
# #         st.info("📚 Agent Action: Executing **Vector Search** for Unstructured Context.")
        
# #         if not vector_search_available:
# #             return "Vector search is not available. Please upload a PDF to create the vector index.", None
        
# #         try:
# #             # 1. Retrieve Context
# #             vector_docs = vector_retriever.invoke(question)
            
# #             if not vector_docs or not vector_docs[0].page_content.strip():
# #                 return "No relevant contextual information found in the documents.", None
            
# #             context = "\n\n".join([doc.page_content for doc in vector_docs])
            
# #             # 2. Show retrieved chunks in expander
# #             with st.expander("📄 Retrieved Context"):
# #                 for i, doc in enumerate(vector_docs, 1):
# #                     st.markdown(f"**Chunk {i}:**")
# #                     st.text(doc.page_content[:300] + "...")
            
# #             # 3. Final Answer Generation (RAG Chain)
# #             rag_prompt = ChatPromptTemplate.from_messages([
# #                 ("system", """You are a professional analyst. Use ONLY the following context to answer the question accurately.
# # Do not invent information. If the context doesn't contain the answer, state that clearly.

# # CONTEXT:
# # {context}"""),
# #                 ("human", "Question: {question}")
# #             ])

# #             rag_chain = (
# #                 RunnableParallel(
# #                     context=lambda x: context,
# #                     question=RunnablePassthrough()
# #                 ) | rag_prompt | llm | StrOutputParser()
# #             )
            
# #             vector_answer = rag_chain.invoke({"question": question})
# #             return vector_answer, None
            
# #         except Exception as e:
# #             return f"Vector search failed: {str(e)}", None

# # # --- STREAMLIT UI ---
# # st.set_page_config(page_title="Advanced Graph RAG Demo", layout="wide")
# # st.title("🧠 Advanced Graph RAG Demo (Gemini + Neo4j)")
# # st.caption("A Hybrid AI Agent for Finance/Inventory Analysis")

# # # --- Sidebar for File Management ---
# # with st.sidebar:
# #     st.header("Upload/Remove Documents")
    
# #     uploaded_file = st.file_uploader("Upload Excel (.xlsx) or PDF (.pdf)", type=['xlsx', 'pdf', 'csv']) # Added CSV support
# #     company_input = st.text_input("Company Name for Tagging:", value="ABC Book Stores")
    
# #     if uploaded_file and st.button("🚀 Ingest & Process Data"):
# #         # Check if the uploaded file is one of the supported types
# #         file_name_lower = uploaded_file.name.lower()
# #         if file_name_lower.endswith(('.xlsx', '.pdf', '.csv')):
# #             file_ext = file_name_lower.split('.')[-1]
# #             # Normalize Excel/CSV to 'xlsx' type for processing utility
# #             file_type = 'xlsx' if file_ext in ['xlsx', 'csv'] else 'pdf' 

# #             with st.spinner(f"Ingesting {uploaded_file.name} (This creates vectors and graph entities)..."):
# #                 # Call the robust ingestion function
# #                 ingestion_status = ingest_data(uploaded_file, file_type, company_input)
# #                 st.session_state['messages'] = []
# #                 st.cache_resource.clear() 
# #                 st.success(ingestion_status)
# #                 st.rerun() 
# #         else:
# #             st.error("Please upload a valid .xlsx, .csv, or .pdf file.")

# #     st.markdown("---")
    
# #     file_to_remove = st.text_input("File Name to Remove:", help="Enter exact file name (e.g., invoice_INV-202510-017.pdf)")
# #     if st.button("🗑️ Remove Data"):
# #         if file_to_remove:
# #             with st.spinner(f"Removing {file_to_remove}..."):
# #                 removal_status = remove_data(file_to_remove)
# #                 st.session_state['messages'] = []
# #                 st.cache_resource.clear()
# #                 st.success(removal_status)
# #                 st.rerun()
# #         else:
# #             st.warning("Please enter a file name.")
            
# #     st.header("ℹ️ Example Queries")
# #     st.markdown("""
# # **Structured (Graph):**
# # - What is the total purchase amount?
# # - Show all book titles
# # - What books are in the Jayanagar store?
# # - List all authors in inventory
# # - What is the GST for invoice INV-202510-017.pdf?

# # **Unstructured (Vector):**
# # - Summarize the purchase orders
# # - What issues were found in the audit?
# # - Explain the key findings.
# # """)

# # # --- Chat Interface ---
# # if "messages" not in st.session_state:
# #     st.session_state.messages = []

# # for message in st.session_state.messages:
# #     with st.chat_message(message["role"]):
# #         st.markdown(message["content"])
# #         if "details" in message:
# #             st.code(message["details"], language='cypher')
# #             st.caption("Cypher Query Used")

# # if prompt := st.chat_input("Ask a question about your data..."):
# #     st.session_state.messages.append({"role": "user", "content": prompt})
# #     with st.chat_message("user"):
# #         st.markdown(prompt)

# #     with st.spinner("🤔 Thinking..."):
# #         try:
# #             final_answer, cypher_query = run_hybrid_query(prompt)
# #         except Exception as e:
# #             final_answer = f"Error: {str(e)}"
# #             cypher_query = None

# #     ai_message = {"role": "assistant", "content": final_answer}
# #     if cypher_query:
# #         ai_message["details"] = cypher_query
    
# #     st.session_state.messages.append(ai_message)
    
# #     with st.chat_message("assistant"):
# #         st.markdown(final_answer)
# #         if cypher_query:
# #             st.code(cypher_query, language='cypher')
# #             st.caption("Cypher Query Used")


# # At lastimport os
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough, RunnableParallel
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.graphs import Neo4jGraph
# from langchain_neo4j import Neo4jVector
# from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

# # Import the custom ingestion logic from your scripts folder
# # NOTE: The helper functions get_llm, get_embeddings, etc., are imported here but defined
# # to be called lazily in the utility file to ensure environment variables are loaded first.
# from scripts.ingest_utility import ingest_data, remove_data, get_llm, get_embeddings, get_neo4j_graph

# # --- 0. Setup and Initialization ---
# load_dotenv()

# # Retrieve credentials and settings from .env
# GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
# NEO4J_URI = os.getenv("NEO4J_URI")
# NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
# LLM_MODEL = os.getenv("LLM_MODEL")
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# # Validate credentials (using the correct variable name)
# if not all([GOOGLE_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
#     st.error("Missing environment variables. Please check your .env file.")
#     st.stop()

# # Cache resources that do not change during runtime
# @st.cache_resource
# def initialize_rag_components():
#     # We use the utility functions to get lazily initialized components
#     llm_instance = get_llm()
#     embeddings_instance = get_embeddings()
#     neo4j_graph_instance = get_neo4j_graph()
    
#     # Ensure the schema is refreshed for the LLM's query generation tool
#     neo4j_graph_instance.refresh_schema()

#     return llm_instance, embeddings_instance, neo4j_graph_instance

# llm, embeddings, neo4j_graph = initialize_rag_components()

# # --- 1. Graph Search (Structured Data Retrieval) ---

# # UPDATED: Cypher prompt includes examples for the new Book/Metric/Store structure
# cypher_generation_prompt = ChatPromptTemplate.from_messages([
#     ("system", """Task: Generate Cypher statement to query a graph database.
# Instructions:
# - Use only the provided relationship types, node labels (e.g., Company, Metric, Book, Store), and properties in the schema.
# - Write only ONE Cypher query with ONE RETURN clause at the end.
# - Be precise with property names and node labels.

# Schema:
# {schema}

# Examples:
# Question: What is the total purchase amount?
# Cypher: MATCH (m:Metric {name: 'Total Purchase Amount'}) RETURN m.value AS total_purchase

# Question: List all book titles
# Cypher: MATCH (b:Book) RETURN b.title AS book_title

# Question: What books are in the Jayanagar store?
# Cypher: MATCH (s:Store {name: 'Bangalore – Jayanagar'})<-[:LOCATED_AT]-(b:Book) RETURN b.title AS book_title

# Question: What is the total opening stock amount?
# Cypher: MATCH (m:Metric {name: 'Total Opening Stock Amount'}) RETURN m.value AS total_opening_stock
# """),
#     ("human", "{question}")
# ])

# graph_qa_chain = GraphCypherQAChain.from_llm(
#     llm=llm,
#     graph=neo4j_graph,
#     verbose=True,
#     return_intermediate_steps=True,
#     allow_dangerous_requests=True,
#     cypher_prompt=cypher_generation_prompt
# )

# # --- 2. Vector Search (Unstructured Data Retrieval) ---

# @st.cache_resource(ttl=3600)
# def setup_vector_retriever():
#     """Initializes and returns the vector retriever for PDF chunks."""
#     try:
#         # This retriever searches the PDF chunks (Node: Chunk)
#         vector_retriever = Neo4jVector.from_existing_index(
#             embeddings,
#             url=NEO4J_URI,
#             username=NEA4J_USERNAME,
#             password=NEO4J_PASSWORD,
#             index_name="pdf_vector_index",
#             node_label="Chunk",
#             text_node_property="text",
#             embedding_node_property="embedding",
#             retrieval_query="""
#                  RETURN node.text AS text, score, 
#                     {source: node.source_file, type: "PDF Chunk"} AS metadata, node.company AS company
#              """
#         ).as_retriever(search_kwargs={"k": 3})
#         return vector_retriever, True
#     except Exception as e:
#         return str(e), False

# vector_retriever, vector_search_available = setup_vector_retriever()


# # --- 3. LLM-Based Query Router (Intelligent Routing) ---

# def intelligent_query_router(question):
#     """Uses LLM to decide between graph (structured) or vector (unstructured) search."""
#     # We use a cleaner version of the prompt that relies on the LLM's understanding
#     # of the core question intent (numerical vs. contextual).
#     router_prompt = ChatPromptTemplate.from_messages([
#         ("system", """You are a precise query routing assistant. Classify the user's question into ONE of TWO categories:

# 1. GRAPH: Questions asking for specific numbers, totals, amounts, lists of inventory, comparisons, or entity relationships (e.g., "What is the total amount?", "List all books", "Which author is linked to which category?").
# 2. VECTOR: Questions asking for summaries, descriptions, explanations, or contextual narrative (e.g., "Summarize the orders from June", "What are the issues?", "Explain the purchase orders").

# Respond with ONLY one word: "GRAPH" or "VECTOR".
# """),
#         ("human", "Question: {question}\n\nClassification:")
#     ])
    
#     router_chain = router_prompt | llm | StrOutputParser()
#     decision = router_chain.invoke({"question": question}).strip().upper()
    
#     # We check if 'GRAPH' is in the decision to be robust against any extra words from the LLM
#     return "GRAPH" if "GRAPH" in decision else "VECTOR"

# # --- 4. Hybrid RAG (The Intelligence Agent) ---

# def run_hybrid_query(question):
#     """Routes queries using LLM-based intelligent routing."""
    
#     route_decision = intelligent_query_router(question)
#     cypher_query = None
    
#     if route_decision == "GRAPH":
#         st.info("🔍 Agent Action: Executing **Graph Search (Cypher)** for Structured/Relational Data.")
        
#         try:
#             # 1. Run the Graph Chain (NL -> Cypher -> Query -> LLM Answer)
#             graph_result = graph_qa_chain.invoke({"query": question})
#             # LangChain returns intermediate steps as a list of dicts. We extract the generated Cypher query.
#             cypher_query = graph_result['intermediate_steps'][0]['query']
            
#             # The result is the final NL answer generated by the LLM
#             return graph_result['result'], cypher_query
            
#         except Exception as e:
#             error_msg = f"Graph search failed (Cypher error or hallucination): {str(e)}\n\nFalling back to Vector Search..."
#             st.warning(error_msg)
#             route_decision = "VECTOR"
#             # Proceed to VECTOR path if graph fails

#     if route_decision == "VECTOR":
#         st.info("📚 Agent Action: Executing **Vector Search** for Unstructured Context.")
        
#         if not vector_search_available:
#             return "Vector search is not available. Please upload a PDF to create the vector index.", None
        
#         try:
#             # 1. Retrieve Context
#             vector_docs = vector_retriever.invoke(question)
            
#             if not vector_docs or not vector_docs[0].page_content.strip():
#                 return "No relevant contextual information found in the documents.", None
            
#             context = "\n\n".join([doc.page_content for doc in vector_docs])
            
#             # 2. Show retrieved chunks in expander
#             with st.expander("📄 Retrieved Context"):
#                 for i, doc in enumerate(vector_docs, 1):
#                     st.markdown(f"**Chunk {i}:**")
#                     st.text(doc.page_content[:300] + "...")
            
#             # 3. Final Answer Generation (RAG Chain)
#             rag_prompt = ChatPromptTemplate.from_messages([
#                 ("system", """You are a professional analyst. Use ONLY the following context to answer the question accurately.
# Do not invent information. If the context doesn't contain the answer, state that clearly.

# CONTEXT:
# {context}"""),
#                 ("human", "Question: {question}")
#             ])

#             rag_chain = (
#                 RunnableParallel(
#                     context=lambda x: context,
#                     question=RunnablePassthrough()
#                 ) | rag_prompt | llm | StrOutputParser()
#             )
            
#             vector_answer = rag_chain.invoke({"question": question})
#             return vector_answer, None
            
#         except Exception as e:
#             return f"Vector search failed: {str(e)}", None

# # --- STREAMLIT UI ---
# st.set_page_config(page_title="Advanced Graph RAG Demo", layout="wide")
# st.title("🧠 Advanced Graph RAG Demo (Gemini + Neo4j)")
# st.caption("A Hybrid AI Agent for Finance/Inventory Analysis")

# # --- Sidebar for File Management ---
# with st.sidebar:
#     st.header("Upload/Remove Documents")
    
#     uploaded_file = st.file_uploader("Upload Excel (.xlsx) or PDF (.pdf)", type=['xlsx', 'pdf', 'csv']) # Added CSV support
#     company_input = st.text_input("Company Name for Tagging:", value="ABC Book Stores")
    
#     if uploaded_file and st.button("🚀 Ingest & Process Data"):
#         # Check if the uploaded file is one of the supported types
#         file_name_lower = uploaded_file.name.lower()
#         if file_name_lower.endswith(('.xlsx', '.pdf', '.csv')):
#             file_name = uploaded_file.name
#             file_ext = file_name_lower.split('.')[-1]
#             # Normalize Excel/CSV to 'xlsx' type for processing utility
#             file_type = 'xlsx' if file_ext in ['xlsx', 'csv'] else 'pdf' 

#             with st.spinner(f"Ingesting {uploaded_file.name} (This creates vectors and graph entities)..."):
#                 # Call the robust ingestion function
#                 ingestion_status = ingest_data(uploaded_file, file_type, company_input)
#                 st.session_state['messages'] = []
#                 st.cache_resource.clear() 
#                 st.success(ingestion_status)
#                 st.rerun() 
#         else:
#             st.error("Please upload a valid .xlsx, .csv, or .pdf file.")

#     st.markdown("---")
    
#     file_to_remove = st.text_input("File Name to Remove:", help="Enter exact file name (e.g., invoice_INV-202510-017.pdf)")
#     if st.button("🗑️ Remove Data"):
#         if file_to_remove:
#             with st.spinner(f"Removing {file_to_remove}..."):
#                 removal_status = remove_data(file_to_remove)
#                 st.session_state['messages'] = []
#                 st.cache_resource.clear()
#                 st.success(removal_status)
#                 st.rerun()
#         else:
#             st.warning("Please enter a file name.")
            
#     st.header("ℹ️ Example Queries")
#     st.markdown("""
# **Structured (Graph):**
# - What is the total purchase amount?
# - Show all book titles
# - What books are in the Jayanagar store?
# - List all authors in inventory
# - What is the GST for invoice INV-202510-017.pdf?

# **Unstructured (Vector):**
# - Summarize the purchase orders
# - What issues were found in the audit?
# - Explain the key findings.
# """)

# # --- Chat Interface ---
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#         if "details" in message:
#             st.code(message["details"], language='cypher')
#             st.caption("Cypher Query Used")

# if prompt := st.chat_input("Ask a question about your data..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.spinner("🤔 Thinking..."):
#         try:
#             final_answer, cypher_query = run_hybrid_query(prompt)
#         except Exception as e:
#             final_answer = f"Error: {str(e)}"
#             cypher_query = None

#     ai_message = {"role": "assistant", "content": final_answer}
#     if cypher_query:
#         ai_message["details"] = cypher_query
    
#     st.session_state.messages.append(ai_message)
    
#     with st.chat_message("assistant"):
#         st.markdown(final_answer)
#         if cypher_query:
#             st.code(cypher_query, language='cypher')
#             st.caption("Cypher Query Used")

# Working
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

from scripts.ingest_utility import ingest_data, remove_data

# --- 0. Setup and Initialization ---
load_dotenv()

GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

if not all([GOOGLE_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    st.error("Missing environment variables. Please check your .env file.")
    st.stop()

@st.cache_resource
def initialize_rag_components():
    llm_instance = ChatGoogleGenerativeAI(model=LLM_MODEL, api_key=GOOGLE_KEY, temperature=0)
    embeddings_instance = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=GOOGLE_KEY)
    neo4j_graph_instance = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )
    neo4j_graph_instance.refresh_schema()
    return llm_instance, embeddings_instance, neo4j_graph_instance

llm, embeddings, neo4j_graph = initialize_rag_components()

# --- 1. Graph Search (Structured Data Retrieval) ---

cypher_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", """Task: Generate Cypher statement to query a graph database.
Instructions:
- Use only the provided relationship types and properties in the schema
- Write only ONE Cypher query with ONE RETURN clause at the end
- Be precise with property names and node labels

Schema:
{schema}

Examples:
Question: What is the Equity value?
Cypher: MATCH (c:Company)-[:HAS_METRIC]->(m:Metric {{name: 'Equity'}}) RETURN m.value AS equity_value

Question: Show all metrics for the company
Cypher: MATCH (c:Company)-[:HAS_METRIC]->(m:Metric) RETURN m.name AS metric_name, m.value AS metric_value

Question: What are the total assets?
Cypher: MATCH (c:Company)-[:HAS_METRIC]->(m:Metric {{name: 'Total Assets'}}) RETURN m.value AS total_assets
"""),
    ("human", "{question}")
])

graph_qa_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=neo4j_graph,
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_requests=True,
    cypher_prompt=cypher_generation_prompt
)

# --- 2. Vector Search (Unstructured Data Retrieval) ---

@st.cache_resource(ttl=3600)
def setup_vector_retriever():
    try:
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
        return vector_retriever, True
    except Exception as e:
        return str(e), False

vector_retriever, vector_search_available = setup_vector_retriever()

# --- 3. LLM-Based Query Router ---

def intelligent_query_router(question):
    """Uses LLM to decide between graph (structured) or vector (unstructured) search."""
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query routing assistant. Your job is to classify user questions into TWO categories:

1. GRAPH - Questions asking for specific data points, numbers, metrics, comparisons, or relationships between entities
   Examples: "What is the equity value?", "Show all metrics", "Compare debt and equity", "List all companies"

2. VECTOR - Questions asking for explanations, summaries, recommendations, or contextual information from documents
   Examples: "What weaknesses were found?", "Summarize the report", "What are the recommendations?", "Explain the findings"

Respond with ONLY one word: either "GRAPH" or "VECTOR"
"""),
        ("human", "Question: {question}\n\nClassification:")
    ])
    
    router_chain = router_prompt | llm | StrOutputParser()
    decision = router_chain.invoke({"question": question}).strip().upper()
    
    return "GRAPH" if "GRAPH" in decision else "VECTOR"

# --- 4. Hybrid RAG (The Intelligence Agent) ---

def run_hybrid_query(question):
    """Routes queries using LLM-based intelligent routing."""
    
    # Use LLM to decide routing
    route_decision = intelligent_query_router(question)
    
    if route_decision == "GRAPH":
        st.info("🔍 Agent Action: Executing **Graph Search (Cypher)** for Structured/Relational Data.")
        
        try:
            graph_result = graph_qa_chain.invoke({"query": question})
            cypher_query = graph_result['intermediate_steps'][0]['query']
            
            st.code(cypher_query, language='cypher')
            st.caption("Generated Cypher Query")
            
            return graph_result['result'], cypher_query
            
        except Exception as e:
            error_msg = f"Graph search failed: {str(e)}\n\nFalling back to Vector Search..."
            st.warning(error_msg)
            route_decision = "VECTOR"
    
    if route_decision == "VECTOR":
        st.info("📚 Agent Action: Executing **Vector Search** for Unstructured Context.")
        
        if not vector_search_available:
            return "Vector search is not available. Please upload a PDF to create the vector index.", None
        
        try:
            vector_docs = vector_retriever.invoke(question)
            
            if not vector_docs or not vector_docs[0].page_content.strip():
                return "No relevant information found in the documents.", None
            
            context = "\n\n".join([doc.page_content for doc in vector_docs])
            
            with st.expander("📄 Retrieved Context"):
                for i, doc in enumerate(vector_docs, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(doc.page_content[:300] + "...")
            
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a professional analyst. Use ONLY the following context to answer the question accurately.
Do not invent information. If the context doesn't contain the answer, state that clearly.

CONTEXT:
{context}"""),
                ("human", "Question: {question}")
            ])

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
st.caption("A Hybrid AI Agent combining Structured Data (Excel metrics) and Unstructured Context (PDF chunks)")

# --- Sidebar for File Management ---
with st.sidebar:
    st.header("Upload/Remove Documents")
    
    uploaded_file = st.file_uploader("Upload Excel (.xlsx) or PDF (.pdf)", type=['xlsx', 'pdf'])
    company_input = st.text_input("Company Name for Tagging:", value="ABC Book Stores")
    
    if uploaded_file and st.button("🚀 Ingest & Process Data"):
        file_ext = uploaded_file.name.split('.')[-1]
        if file_ext in ['xlsx', 'pdf']:
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                ingestion_status = ingest_data(uploaded_file, file_ext, company_input)
                st.session_state['messages'] = []
                st.cache_resource.clear()
                st.success(ingestion_status)
                st.rerun()
        else:
            st.error("Please upload a valid .xlsx or .pdf file.")

    st.markdown("---")
    
    file_to_remove = st.text_input("File Name to Remove:", help="Enter exact file name")
    if st.button("🗑️ Remove Data"):
        if file_to_remove:
            with st.spinner(f"Removing {file_to_remove}..."):
                removal_status = remove_data(file_to_remove)
                st.session_state['messages'] = []
                st.cache_resource.clear()
                st.success(removal_status)
                st.rerun()
        else:
            st.warning("Please enter a file name.")
            
    st.header("ℹ️ Example Queries")
    st.markdown("""
**Graph Search (Structured Data):**
- What is the total opening stock amount?
- Show all book titles
- What books are in Jayanagar store?
- Who wrote The Alchemist?
- List all publishers

**Vector Search (PDF Context):**
- Summarize the invoice details
- What items were sold in June?
- Describe the payment terms
- Explain the bank information
""")
    
    # st.markdown("---")
    # st.header("📊 Database Status")

    # if st.button("Show What's Stored"):
    #     driver = get_db_driver()
    #     try:
    #         with driver.session() as session:
    #             # Count by node type
    #             result = session.run("""
    #                 MATCH (n)
    #                 RETURN labels(n)[0] AS type, count(n) AS count
    #                 ORDER BY count DESC
    #             """)
                
    #             st.write("**Nodes by Type:**")
    #             for record in result:
    #                 st.write(f"- {record['type']}: {record['count']}")
                
    #             # Show files
    #             result = session.run("""
    #                 MATCH (n)
    #                 WHERE n.source_file IS NOT EXISTS(n.source_file) = false
    #                 RETURN DISTINCT n.source_file AS file
    #             """)
                
    #             files = [record['file'] for record in result]
    #             st.write(f"\n**Files Loaded ({len(files)}):**")
    #             for f in files:
    #                 st.write(f"- {f}")
                
    #             # Check vector index
    #             result = session.run("CALL db.indexes() YIELD name WHERE name = 'pdf_vector_index' RETURN count(*) as has_vector")
    #             has_vector = list(result)[0]['has_vector'] > 0
    #             st.write(f"\n**Vector Search:** {'Available' if has_vector else 'Not Available'}")
    #     finally:
    #         driver.close()
    
# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "details" in message:
            st.code(message["details"], language='cypher')
            st.caption("Cypher Query Used")

if prompt := st.chat_input("Ask a question about your data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("🤔 Thinking..."):
        try:
            final_answer, cypher_query = run_hybrid_query(prompt)
        except Exception as e:
            final_answer = f"Error: {str(e)}"
            cypher_query = None

    ai_message = {"role": "assistant", "content": final_answer}
    if cypher_query:
        ai_message["details"] = cypher_query
    
    st.session_state.messages.append(ai_message)
    
    with st.chat_message("assistant"):
        st.markdown(final_answer)
        if cypher_query:
            st.code(cypher_query, language='cypher')
            st.caption("Cypher Query Used")