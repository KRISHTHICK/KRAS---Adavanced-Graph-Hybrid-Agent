# # # # # # # kras/scripts/ingest_utility.py
# # # # # # import os
# # # # # # import pandas as pd
# # # # # # from io import BytesIO
# # # # # # from neo4j import GraphDatabase, exceptions
# # # # # # from langchain_community.document_loaders import PDFPlumberLoader
# # # # # # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # # # # # from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# # # # # # from langchain_neo4j import Neo4jVector, Neo4jGraph
# # # # # # from langchain_experimental.graph_transformers import LLMGraphTransformer
# # # # # # from langchain_core.documents import Document

# # # # # # # --- 0. Setup ---
# # # # # # # Credentials are loaded in app.py, so we pull them from the OS environment
# # # # # # LLM_MODEL = os.getenv("LLM_MODEL")
# # # # # # EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
# # # # # # GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
# # # # # # NEO4J_URI = os.getenv("NEO4J_URI")
# # # # # # NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# # # # # # NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# # # # # # # Initialize models once
# # # # # # llm = ChatGoogleGenerativeAI(model=LLM_MODEL, api_key=GOOGLE_KEY, temperature=0)
# # # # # # embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=GOOGLE_KEY)
# # # # # # neo4j_graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# # # # # # def clear_data_by_source(file_name):
# # # # # #     """Deletes all nodes/relationships associated with a specific file_name."""
# # # # # #     driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
# # # # # #     delete_query = """
# # # # # #     MATCH (n) WHERE n.source_file = $file_name OR n.source_file = $file_name_short 
# # # # # #     DETACH DELETE n
# # # # # #     """
# # # # # #     with driver.session() as session:
# # # # # #         result = session.run(delete_query, file_name=file_name, file_name_short=file_name.split('/')[-1])
# # # # # #         return result.consume().counters.nodes_deleted

# # # # # # def ingest_data(uploaded_file, file_type, company_name):
# # # # # #     """Handles file ingestion into Neo4j (Structured and Unstructured paths)."""
# # # # # #     file_name = uploaded_file.name
    
# # # # # #     # 1. Clear existing data for this file
# # # # # #     deleted_count = clear_data_by_source(file_name)

# # # # # #     # --- Excel/Structured Data Path ---
# # # # # #     if file_type == 'xlsx':
# # # # # #         # Read file directly from the Streamlit UploadedFile buffer
# # # # # #         df = pd.read_excel(uploaded_file, engine='openpyxl')
        
# # # # # #         # In a real app, logic extracts these dynamically, here we mock based on assumption
# # # # # #         financial_metrics = {
# # # # # #             "Equity": df.iloc[0, 1] if df.shape[0]>0 and df.shape[1]>1 else 1200000,
# # # # # #             "Debt": df.iloc[1, 1] if df.shape[0]>1 and df.shape[1]>1 else 500000,
# # # # # #             "TaxRate": df.iloc[2, 1] if df.shape[0]>2 and df.shape[1]>1 else 0.21
# # # # # #         }
        
# # # # # #         def create_excel_metrics(tx):
# # # # # #             # Create Company Node (Use MERGE to avoid duplicates if company is already there from a PDF)
# # # # # #             tx.run("MERGE (c:Company {name: $name}) ON CREATE SET c.cik='XYZ123'", name=company_name)
            
# # # # # #             for name, value in financial_metrics.items():
# # # # # #                 tx.run("""
# # # # # #                     MATCH (c:Company {name: $c_name}) 
# # # # # #                     MERGE (m:Metric {name: $m_name, source_file: $f_name}) 
# # # # # #                     ON CREATE SET m.value = $m_value
# # # # # #                     ON MATCH SET m.value = $m_value
# # # # # #                     MERGE (c)-[:HAS_METRIC]->(m)
# # # # # #                 """, c_name=company_name, m_name=name, m_value=value, f_name=file_name)
        
# # # # # #         driver = get_db_driver()
# # # # # #         with driver.session() as session:
# # # # # #             session.execute_write(create_excel_metrics)
# # # # # #         return f"✅ {deleted_count} old nodes deleted. Structured metrics loaded from {file_name}."

# # # # # #     # --- PDF/Hybrid Data Path (Vector + Structured Graph) ---
# # # # # #     elif file_type == 'pdf':
# # # # # #         # 1. Load Documents (Save to temp file to enable LangChain PDFLoader)
# # # # # #         temp_path = os.path.join(os.getcwd(), uploaded_file.name)
# # # # # #         with open(temp_path, "wb") as f:
# # # # # #             f.write(uploaded_file.getbuffer())
        
# # # # # #         loader = PDFPlumberLoader(temp_path)
# # # # # #         documents = loader.load()
        
# # # # # #         # 2. Chunking for Vectors (The Unstructured Part)
# # # # # #         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
# # # # # #         chunks = text_splitter.split_documents(documents)
# # # # # #         for chunk in chunks:
# # # # # #             chunk.metadata['source_file'] = file_name
        
# # # # # #         # 3. Vector Indexing (The Unstructured Part)
# # # # # #         vector_db = Neo4jVector.from_documents(
# # # # # #             chunks, embeddings, url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
# # # # # #             index_name="pdf_vector_index", node_label="Chunk"
# # # # # #         )
        
# # # # # #         # 4. LLM Graph Structuring (The Structured Part)
# # # # # #         # Use LLM Graph Transformer to extract key entities and relationships from PDF text
# # # # # #         llm_transformer = LLMGraphTransformer(
# # # # # #             llm=llm,
# # # # # #             allowed_nodes=['Company', 'Person', 'Concept', 'Risk'],
# # # # # #             allowed_relationships=['HAS_RISK', 'DISCUSSED', 'IDENTIFIES']
# # # # # #         )
        
# # # # # #         # This takes chunks and asks Gemini to structure the data into graph objects
# # # # # #         graph_documents = llm_transformer.convert_to_graph_documents(chunks[:5]) # Limit to first 5 chunks for speed
# # # # # #         neo4j_graph.add_graph_documents(graph_documents)

# # # # # #         # 5. Clean up temp file
# # # # # #         os.remove(temp_path)
        
# # # # # #         return f"✅ {deleted_count} old nodes deleted. {len(chunks)} Chunks and {len(graph_documents)} new graph entities loaded from {file_name}."
    
# # # # # #     return "Unsupported file type."

# # # # # # def remove_data(file_name):
# # # # # #     """Removes all nodes and relationships related to a specific file."""
# # # # # #     deleted_count = clear_data_by_source(file_name)
# # # # # #     return f"🗑️ Successfully deleted {deleted_count} nodes and related data for {file_name}."

# # # # # # PLan 2
# # # # # # kras/scripts/ingest_utility.py
# # # # # import os
# # # # # import pandas as pd
# # # # # from neo4j import GraphDatabase
# # # # # from langchain_community.document_loaders import PDFPlumberLoader
# # # # # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # # # # from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# # # # # from langchain_neo4j import Neo4jVector
# # # # # from langchain_community.graphs import Neo4jGraph
# # # # # from langchain_experimental.graph_transformers import LLMGraphTransformer
# # # # # from langchain_core.documents import Document

# # # # # # DON'T initialize these at module level - they'll be None
# # # # # # llm = ChatGoogleGenerativeAI(...)  # REMOVE THIS

# # # # # def get_llm():
# # # # #     """Lazy initialization of LLM"""
# # # # #     LLM_MODEL = os.getenv("LLM_MODEL")
# # # # #     GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
# # # # #     return ChatGoogleGenerativeAI(model=LLM_MODEL, api_key=GOOGLE_KEY, temperature=0)

# # # # # def get_embeddings():
# # # # #     """Lazy initialization of embeddings"""
# # # # #     EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
# # # # #     GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
# # # # #     return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=GOOGLE_KEY)

# # # # # def get_neo4j_graph():
# # # # #     """Lazy initialization of Neo4j graph"""
# # # # #     NEO4J_URI = os.getenv("NEO4J_URI")
# # # # #     NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# # # # #     NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
# # # # #     return Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# # # # # def get_db_driver():
# # # # #     """Returns a Neo4j Driver instance."""
# # # # #     NEO4J_URI = os.getenv("NEO4J_URI")
# # # # #     NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# # # # #     NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
# # # # #     return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# # # # # def clear_data_by_source(file_name):
# # # # #     """Deletes all nodes/relationships associated with a specific file_name."""
# # # # #     driver = get_db_driver()
# # # # #     delete_query = """
# # # # #     MATCH (n) 
# # # # #     WHERE n.source_file = $file_name OR n.file_name = $file_name
# # # # #     DETACH DELETE n
# # # # #     """
# # # # #     with driver.session() as session:
# # # # #         result = session.run(delete_query, file_name=file_name)
# # # # #         return result.consume().counters.nodes_deleted

# # # # # def ingest_data(uploaded_file, file_type, company_name):
# # # # #     """Processes an uploaded file and loads data into Neo4j."""
# # # # #     file_name = uploaded_file.name
    
# # # # #     # Get credentials
# # # # #     NEO4J_URI = os.getenv("NEO4J_URI")
# # # # #     NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# # # # #     NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    
# # # # #     # 1. Clear existing data
# # # # #     deleted_count = clear_data_by_source(file_name)

# # # # #     # --- Excel Path ---
# # # # #     if file_type == 'xlsx':
# # # # #         try:
# # # # #             df = pd.read_excel(uploaded_file, engine='openpyxl')
# # # # #         except Exception as e:
# # # # #             return f"❌ Error reading Excel: {e}"

# # # # #         # Extract metrics (adjust based on your Excel structure)
# # # # #         financial_metrics = {}
# # # # #         for idx, row in df.iterrows():
# # # # #             if len(row) >= 2:
# # # # #                 metric_name = str(row[0]).strip()
# # # # #                 metric_value = row[1]
# # # # #                 if metric_name and metric_name != 'nan':
# # # # #                     financial_metrics[metric_name] = metric_value
        
# # # # #         def create_excel_metrics(tx):
# # # # #             tx.run("MERGE (c:Company {name: $name})", name=company_name)
            
# # # # #             for name, value in financial_metrics.items():
# # # # #                 tx.run("""
# # # # #                     MATCH (c:Company {name: $c_name}) 
# # # # #                     MERGE (m:Metric {name: $m_name, source_file: $f_name}) 
# # # # #                     SET m.value = $m_value
# # # # #                     MERGE (c)-[:HAS_METRIC]->(m)
# # # # #                 """, c_name=company_name, m_name=name, m_value=value, f_name=file_name)

# # # # #         driver = get_db_driver()
# # # # #         with driver.session() as session:
# # # # #             session.execute_write(create_excel_metrics)
        
# # # # #         return f"✅ Deleted {deleted_count} old nodes. Loaded {len(financial_metrics)} metrics from {file_name}."

# # # # #     # --- PDF Path ---
# # # # #     elif file_type == 'pdf':
# # # # #         # Save temp file
# # # # #         temp_path = os.path.join(os.getcwd(), "data_sources", file_name)
# # # # #         os.makedirs("data_sources", exist_ok=True)
        
# # # # #         with open(temp_path, "wb") as f:
# # # # #             f.write(uploaded_file.getbuffer())
        
# # # # #         # Load and chunk
# # # # #         loader = PDFPlumberLoader(temp_path)
# # # # #         documents = loader.load()
        
# # # # #         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
# # # # #         chunks = text_splitter.split_documents(documents)
        
# # # # #         for chunk in chunks:
# # # # #             chunk.metadata['source_file'] = file_name
        
# # # # #         # Create vector index
# # # # #         embeddings = get_embeddings()
# # # # #         vector_db = Neo4jVector.from_documents(
# # # # #             chunks, 
# # # # #             embeddings, 
# # # # #             url=NEO4J_URI, 
# # # # #             username=NEO4J_USERNAME, 
# # # # #             password=NEO4J_PASSWORD,
# # # # #             index_name="pdf_vector_index", 
# # # # #             node_label="Chunk",
# # # # #             text_node_property="text",
# # # # #             embedding_node_property="embedding"
# # # # #         )
        
# # # # #         # Optional: Extract entities with LLM (this is slow, consider removing for speed)
# # # # #         try:
# # # # #             llm = get_llm()
# # # # #             neo4j_graph = get_neo4j_graph()
            
# # # # #             llm_transformer = LLMGraphTransformer(
# # # # #                 llm=llm,
# # # # #                 allowed_nodes=['Company', 'Person', 'Concept', 'Risk'],
# # # # #                 allowed_relationships=['HAS_RISK', 'DISCUSSED', 'IDENTIFIES']
# # # # #             )
            
# # # # #             # Only process first 5 chunks for speed
# # # # #             graph_documents = llm_transformer.convert_to_graph_documents(chunks[:5])
# # # # #             neo4j_graph.add_graph_documents(graph_documents)
# # # # #         except Exception as e:
# # # # #             print(f"Warning: Entity extraction failed: {e}")
        
# # # # #         return f"✅ Deleted {deleted_count} old nodes. Loaded {len(chunks)} chunks from {file_name}."
    
# # # # #     return "❌ Unsupported file type."

# # # # # def remove_data(file_name):
# # # # #     """Removes all nodes and relationships related to a specific file."""
# # # # #     deleted_count = clear_data_by_source(file_name)
# # # # #     return f"🗑️ Deleted {deleted_count} nodes for {file_name}."

# # # # # PLAN 3
# # # # # kras/scripts/ingest_utility.py
# # # # import os
# # # # import pandas as pd
# # # # from neo4j import GraphDatabase
# # # # from langchain_community.document_loaders import PDFPlumberLoader
# # # # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # # # from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# # # # from langchain_neo4j import Neo4jVector
# # # # from langchain_community.graphs import Neo4jGraph
# # # # from langchain_experimental.graph_transformers import LLMGraphTransformer

# # # # def get_llm():
# # # #     """Lazy initialization of LLM"""
# # # #     LLM_MODEL = os.getenv("LLM_MODEL")
# # # #     GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
# # # #     return ChatGoogleGenerativeAI(model=LLM_MODEL, api_key=GOOGLE_KEY, temperature=0)

# # # # def get_embeddings():
# # # #     """Lazy initialization of embeddings"""
# # # #     EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
# # # #     GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
# # # #     return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=GOOGLE_KEY)

# # # # def get_neo4j_graph():
# # # #     """Lazy initialization of Neo4j graph"""
# # # #     NEO4J_URI = os.getenv("NEO4J_URI")
# # # #     NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# # # #     NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
# # # #     return Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# # # # def get_db_driver():
# # # #     """Returns a Neo4j Driver instance"""
# # # #     NEO4J_URI = os.getenv("NEO4J_URI")
# # # #     NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# # # #     NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
# # # #     return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# # # # def clear_data_by_source(file_name):
# # # #     """Deletes all nodes/relationships associated with a specific file_name"""
# # # #     driver = get_db_driver()
# # # #     delete_query = """
# # # #     MATCH (n) 
# # # #     WHERE n.source_file = $file_name OR n.file_name = $file_name
# # # #     DETACH DELETE n
# # # #     """
# # # #     with driver.session() as session:
# # # #         result = session.run(delete_query, file_name=file_name)
# # # #         deleted = result.consume().counters.nodes_deleted
# # # #         driver.close()
# # # #         return deleted

# # # # def ingest_data(uploaded_file, file_type, company_name):
# # # #     """Processes uploaded file and loads into Neo4j (Hybrid Structured + Vector)"""
# # # #     file_name = uploaded_file.name
    
# # # #     NEO4J_URI = os.getenv("NEO4J_URI")
# # # #     NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# # # #     NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    
# # # #     deleted_count = clear_data_by_source(file_name)

# # # #     # --- Excel/Structured Data Path ---
# # # #     if file_type == 'xlsx':
# # # #         try:
# # # #             df = pd.read_excel(uploaded_file, engine='openpyxl')
# # # #         except Exception as e:
# # # #             return f"❌ Error reading Excel: {e}"

# # # #         # Smart extraction: treat first column as metric name, second as value
# # # #         financial_metrics = {}
# # # #         for idx, row in df.iterrows():
# # # #             if len(row) >= 2:
# # # #                 metric_name = str(row[0]).strip()
# # # #                 metric_value = row[1]
                
# # # #                 # Skip empty or invalid rows
# # # #                 if metric_name and metric_name.lower() not in ['nan', 'none', '']:
# # # #                     try:
# # # #                         # Try to convert to number if possible
# # # #                         if isinstance(metric_value, str):
# # # #                             metric_value = float(metric_value.replace(',', ''))
# # # #                         financial_metrics[metric_name] = metric_value
# # # #                     except:
# # # #                         financial_metrics[metric_name] = metric_value
        
# # # #         if not financial_metrics:
# # # #             return "❌ No valid metrics found in Excel file. Check format."
        
# # # #         def create_excel_metrics(tx):
# # # #             # Create or merge company node
# # # #             tx.run("MERGE (c:Company {name: $name})", name=company_name)
            
# # # #             # Create metrics and relationships
# # # #             for name, value in financial_metrics.items():
# # # #                 tx.run("""
# # # #                     MATCH (c:Company {name: $c_name}) 
# # # #                     MERGE (m:Metric {name: $m_name, source_file: $f_name}) 
# # # #                     SET m.value = $m_value
# # # #                     MERGE (c)-[:HAS_METRIC]->(m)
# # # #                 """, c_name=company_name, m_name=name, m_value=value, f_name=file_name)
            
# # # #             # Create document node for metadata
# # # #             tx.run("""
# # # #                 MERGE (d:Document {file_name: $f_name, type: 'Excel'})
# # # #                 SET d.title = $title
# # # #                 WITH d
# # # #                 MATCH (c:Company {name: $c_name})
# # # #                 MERGE (c)-[:FILED]->(d)
# # # #             """, f_name=file_name, title=f"Metrics from {file_name}", c_name=company_name)

# # # #         driver = get_db_driver()
# # # #         try:
# # # #             with driver.session() as session:
# # # #                 session.execute_write(create_excel_metrics)
# # # #         finally:
# # # #             driver.close()
        
# # # #         return f"✅ Deleted {deleted_count} old nodes. Loaded {len(financial_metrics)} metrics from {file_name}."

# # # #     # --- PDF/Hybrid Data Path (Vector + Structured Graph) ---
# # # #     elif file_type == 'pdf':
# # # #         # Save temp file for processing
# # # #         temp_dir = "data_sources"
# # # #         os.makedirs(temp_dir, exist_ok=True)
# # # #         temp_path = os.path.join(temp_dir, file_name)
        
# # # #         with open(temp_path, "wb") as f:
# # # #             f.write(uploaded_file.getbuffer())
        
# # # #         try:
# # # #             # Load and parse PDF
# # # #             loader = PDFPlumberLoader(temp_path)
# # # #             documents = loader.load()
            
# # # #             if not documents:
# # # #                 return "❌ No content extracted from PDF."
            
# # # #             # Chunking for vector search
# # # #             text_splitter = RecursiveCharacterTextSplitter(
# # # #                 chunk_size=1024, 
# # # #                 chunk_overlap=100,
# # # #                 separators=["\n\n", "\n", ". ", " ", ""]
# # # #             )
# # # #             chunks = text_splitter.split_documents(documents)
            
# # # #             # Tag chunks with source file
# # # #             for chunk in chunks:
# # # #                 chunk.metadata['source_file'] = file_name
# # # #                 chunk.metadata['company'] = company_name
            
# # # #             # Create vector index
# # # #             embeddings = get_embeddings()
            
# # # #             vector_db = Neo4jVector.from_documents(
# # # #                 chunks, 
# # # #                 embeddings, 
# # # #                 url=NEO4J_URI, 
# # # #                 username=NEO4J_USERNAME, 
# # # #                 password=NEO4J_PASSWORD,
# # # #                 index_name="pdf_vector_index", 
# # # #                 node_label="Chunk",
# # # #                 text_node_property="text",
# # # #                 embedding_node_property="embedding",
# # # #                 retrieval_query="""
# # # #                     RETURN node.text AS text, score, 
# # # #                     {source: node.source_file, type: "PDF Chunk"} AS metadata
# # # #                 """
# # # #             )
            
# # # #             # Optional: Extract entities with LLM (can be slow, limit to 5 chunks)
# # # #             entity_extraction_enabled = True
            
# # # #             if entity_extraction_enabled and len(chunks) > 0:
# # # #                 try:
# # # #                     llm = get_llm()
# # # #                     neo4j_graph = get_neo4j_graph()
                    
# # # #                     llm_transformer = LLMGraphTransformer(
# # # #                         llm=llm,
# # # #                         allowed_nodes=['Company', 'Person', 'Product', 'Concept', 'Risk', 'Finding'],
# # # #                         allowed_relationships=['MENTIONS', 'HAS_RISK', 'DISCUSSES', 'RECOMMENDS']
# # # #                     )
                    
# # # #                     # Process first 3 chunks only for speed
# # # #                     sample_chunks = chunks[:3]
# # # #                     graph_documents = llm_transformer.convert_to_graph_documents(sample_chunks)
                    
# # # #                     if graph_documents:
# # # #                         neo4j_graph.add_graph_documents(graph_documents)
                        
# # # #                         # Link entities to company
# # # #                         driver = get_db_driver()
# # # #                         try:
# # # #                             with driver.session() as session:
# # # #                                 session.run("""
# # # #                                     MATCH (c:Company {name: $c_name})
# # # #                                     MATCH (e) 
# # # #                                     WHERE (e:Concept OR e:Risk OR e:Finding) 
# # # #                                       AND e.source_file = $f_name
# # # #                                     MERGE (c)-[:RELATED_TO]->(e)
# # # #                                 """, c_name=company_name, f_name=file_name)
# # # #                         finally:
# # # #                             driver.close()
                        
# # # #                 except Exception as e:
# # # #                     print(f"Warning: Entity extraction failed: {e}")
            
# # # #             # Create document metadata node
# # # #             driver = get_db_driver()
# # # #             try:
# # # #                 with driver.session() as session:
# # # #                     session.run("""
# # # #                         MERGE (d:Document {file_name: $f_name, type: 'PDF'})
# # # #                         SET d.title = $title, d.pages = $pages
# # # #                         WITH d
# # # #                         MATCH (c:Company {name: $c_name})
# # # #                         MERGE (c)-[:FILED]->(d)
# # # #                     """, f_name=file_name, title=file_name, pages=len(documents), c_name=company_name)
# # # #             finally:
# # # #                 driver.close()
            
# # # #             return f"✅ Deleted {deleted_count} old nodes. Loaded {len(chunks)} chunks from {file_name} (Vector + Graph entities)."
            
# # # #         except Exception as e:
# # # #             return f"❌ Error processing PDF: {e}"
        
# # # #         finally:
# # # #             # Clean up temp file
# # # #             if os.path.exists(temp_path):
# # # #                 try:
# # # #                     os.remove(temp_path)
# # # #                 except:
# # # #                     pass
    
# # # #     return "❌ Unsupported file type."

# # # # def remove_data(file_name):
# # # #     """Removes all nodes and relationships related to a specific file"""
# # # #     deleted_count = clear_data_by_source(file_name)
    
# # # #     if deleted_count > 0:
# # # #         return f"🗑️ Successfully deleted {deleted_count} nodes for {file_name}."
# # # #     else:
# # # #         return f"⚠️ No data found for {file_name}. Check the filename."

# # # #Plan final
# # # import os
# # # import pandas as pd
# # # from neo4j import GraphDatabase, exceptions
# # # from langchain_community.document_loaders import PDFPlumberLoader
# # # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # # from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# # # from langchain_neo4j import Neo4jVector
# # # from langchain_community.graphs import Neo4jGraph
# # # from langchain_experimental.graph_transformers import LLMGraphTransformer

# # # # --- 0. Setup Utilities ---

# # # def get_llm():
# # #     """Lazy initialization of LLM"""
# # #     LLM_MODEL = os.getenv("LLM_MODEL")
# # #     GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
# # #     return ChatGoogleGenerativeAI(model=LLM_MODEL, api_key=GOOGLE_KEY, temperature=0)

# # # def get_embeddings():
# # #     """Lazy initialization of embeddings"""
# # #     EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
# # #     GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
# # #     return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=GOOGLE_KEY)

# # # def get_neo4j_graph():
# # #     """Lazy initialization of Neo4j graph"""
# # #     NEO4J_URI = os.getenv("NEO4J_URI")
# # #     NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# # #     NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
# # #     return Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# # # # FIX: Corrected the typo in URI variable name (NEA4J_URI -> NEO4J_URI)
# # # def get_db_driver():
# # #     """Returns a Neo4j Driver instance"""
# # #     NEO4J_URI = os.getenv("NEO4J_URI")
# # #     NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# # #     NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
# # #     return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# # # def clear_data_by_source(file_name):
# # #     """Deletes all nodes/relationships associated with a specific file_name"""
# # #     driver = get_db_driver()
# # #     delete_query = """
# # #     MATCH (n) 
# # #     WHERE n.source_file = $file_name OR n.file_name = $file_name
# # #     DETACH DELETE n
# # #     """
# # #     with driver.session() as session:
# # #         result = session.run(delete_query, file_name=file_name)
# # #         deleted = result.consume().counters.nodes_deleted
# # #         driver.close()
# # #         return deleted

# # # def ingest_data(uploaded_file, file_type, company_name):
# # #     """Processes uploaded file and loads into Neo4j (Hybrid Structured + Vector)"""
# # #     file_name = uploaded_file.name
    
# # #     NEO4J_URI = os.getenv("NEO4J_URI")
# # #     NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# # #     NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    
# # #     deleted_count = clear_data_by_source(file_name)

# # #     # --- Excel/Structured Data Path (Tabular Data Modeling) ---
# # #     if file_type == 'xlsx':
# # #         try:
# # #             # Read all sheets, then process the main inventory sheet
# # #             xlsx = pd.ExcelFile(uploaded_file)
# # #             df = xlsx.parse(sheet_name='Inventory Register')
            
# # #             # Assuming row 1 contains the master metrics (Stock, Purchase)
# # #             # Assuming row 2 contains the column headers for the book inventory
# # #             # Adjusting indexing for pandas internal 0-based indexing on the full sheet data
# # #             master_metrics_df = df.iloc[0:2] # Top metrics and totals
# # #             books_inventory_df = df.iloc[2:] # Inventory records start from row 3 (index 2)
            
# # #             # Clean up column headers using row 2 (index 1) which contains meaningful names
# # #             books_inventory_df.columns = df.iloc[1] 
# # #             books_inventory_df = books_inventory_df.rename(columns={books_inventory_df.columns[0]: 'Category'}) # Rename first column
# # #             books_inventory_df = books_inventory_df.reset_index(drop=True)
            
# # #         except Exception as e:
# # #             return f"❌ Error reading Excel. Check file format/sheet name ('Inventory Register'). Error: {e}"

# # #         def create_excel_metrics(tx):
# # #             # 1. Create or MERGE Company node
# # #             tx.run("MERGE (c:Company {name: $name})", name=company_name)
            
# # #             # 2. Ingest MASTER Metrics (e.g., Totals/Rates from the top rows)
# # #             master_metrics_data = {
# # #                 'Total Opening Stock Amount': master_metrics_df.iloc[0, 1] if master_metrics_df.shape[1] > 1 else 0,
# # #                 'Total Purchase Amount': master_metrics_df.iloc[1, 1] if master_metrics_df.shape[1] > 1 else 0,
# # #                 'Total Sales Amount': master_metrics_df.iloc[2, 1] if master_metrics_df.shape[1] > 1 else 0,
# # #             }
            
# # #             for name, value in master_metrics_data.items():
# # #                  tx.run("""
# # #                     MATCH (c:Company {name: $c_name}) 
# # #                     MERGE (m:Metric {name: $m_name, source_file: $f_name}) 
# # #                     SET m.value = $m_value
# # #                     MERGE (c)-[:HAS_METRIC]->(m)
# # #                 """, c_name=company_name, m_name=name, m_value=value, f_name=file_name)

# # #             # 3. Ingest BOOKS and INVOICES (Row-based entities)
# # #             book_count = 0
# # #             for idx, row in books_inventory_df.iterrows():
# # #                 try:
# # #                     book_title = str(row.get('Book Title', f"Book {idx}")).strip()
# # #                     if book_title == 'nan' or not book_title: continue
                    
# # #                     # Store data fields as properties on the Book node
# # #                     tx.run("""
# # #                         MATCH (c:Company {name: $c_name})
# # #                         MERGE (b:Book {title: $title, source_file: $f_name})
# # #                         ON CREATE SET b.author = $author, b.isbn = $isbn, b.category = $cat, 
# # #                                 b.opening_units = toInteger($op_u), b.purchase_units = toInteger($pur_u)
# # #                         ON MATCH SET b.opening_units = toInteger($op_u), b.purchase_units = toInteger($pur_u)
# # #                         MERGE (c)-[:SELLS]->(b)
# # #                     """, c_name=company_name, title=book_title, f_name=file_name,
# # #                         author=str(row.get('Author', 'Unknown')), isbn=str(row.get('ISBN', 'N/A')),
# # #                         cat=str(row.get('Category', 'Unknown')),
# # #                         op_u=row.get('Opening Stock (Units)', 0),
# # #                         pur_u=row.get('Purchase Units', 0))

# # #                     # Link to Store Location (if column exists)
# # #                     if 'Store Location' in row and str(row['Store Location']).strip() != 'nan':
# # #                         store_name = str(row['Store Location']).strip()
# # #                         tx.run("""
# # #                             MERGE (s:Store {name: $store_name})
# # #                             MERGE (b:Book {title: $title, source_file: $f_name})
# # #                             MERGE (b)-[:LOCATED_AT]->(s)
# # #                         """, store_name=store_name, title=book_title, f_name=file_name)

# # #                     book_count += 1
# # #                 except Exception as e:
# # #                     # In a hackathon, we skip bad rows but log the issue
# # #                     print(f"Skipping row {idx} due to critical error during book processing: {e}")
            
# # #             # Create document metadata node
# # #             tx.run("""
# # #                 MERGE (d:Document {file_name: $f_name, type: 'Excel_Inventory'})
# # #                 ON CREATE SET d.title = $title
# # #                 WITH d
# # #                 MATCH (c:Company {name: $c_name})
# # #                 MERGE (c)-[:FILED]->(d)
# # #             """, f_name=file_name, title=f"Inventory for {file_name}", c_name=company_name)
            
# # #             return book_count

# # #         driver = get_db_driver()
# # #         try:
# # #             with driver.session() as session:
# # #                 book_count = session.execute_write(create_excel_metrics)
# # #         finally:
# # #             driver.close()
        
# # #         return f"✅ Deleted {deleted_count} old nodes. Loaded {book_count} Books and Metrics from {file_name}."

# # #     # --- PDF/Hybrid Data Path (Vector + Structured Graph) ---
# # #     elif file_type == 'pdf':
        
# # #         # Save temp file for processing
# # #         temp_dir = "data_sources"
# # #         os.makedirs(temp_dir, exist_ok=True)
# # #         temp_path = os.path.join(temp_dir, file_name)
        
# # #         with open(temp_path, "wb") as f:
# # #             f.write(uploaded_file.getbuffer())
        
# # #         try:
# # #             loader = PDFPlumberLoader(temp_path)
# # #             documents = loader.load()
            
# # #             if not documents: return "❌ No content extracted from PDF."
            
# # #             # Chunking for vector search
# # #             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
# # #             chunks = text_splitter.split_documents(documents)
            
# # #             for chunk in chunks:
# # #                 chunk.metadata['source_file'] = file_name 
# # #                 chunk.metadata['company'] = company_name
            
# # #             # 3. Vector Indexing (The Unstructured Part)
# # #             embeddings_instance = get_embeddings()
# # #             vector_db = Neo4jVector.from_documents(
# # #                 chunks, embeddings_instance, url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
# # #                 index_name="pdf_vector_index", node_label="Chunk"
# # #             )
            
# # #             # 4. LLM Graph Structuring (The Structured/Relational Part from PDF Text)
# # #             entity_extraction_enabled = True
            
# # #             if entity_extraction_enabled and len(chunks) > 0:
# # #                 try:
# # #                     llm_instance = get_llm()
# # #                     neo4j_graph_instance = get_neo4j_graph()
                    
# # #                     llm_transformer = LLMGraphTransformer(
# # #                         llm=llm_instance,
# # #                         allowed_nodes=['Company', 'Person', 'Product', 'Concept', 'Finding', 'Invoice', 'Buyer'],
# # #                         allowed_relationships=['MENTIONS', 'HAS_RISK', 'DISCUSSES', 'ISSUED_TO', 'CONTAINS_ITEM']
# # #                     )
                    
# # #                     # Process first 5 chunks only for speed
# # #                     sample_chunks = chunks[:5]
# # #                     graph_documents = llm_transformer.convert_to_graph_documents(sample_chunks)
                    
# # #                     if graph_documents:
# # #                         neo4j_graph_instance.add_graph_documents(graph_documents)

# # #                     # Link entities to company (Crucial for multi-hop queries)
# # #                     driver = get_db_driver()
# # #                     with driver.session() as session:
# # #                          session.run("""
# # #                             MERGE (c:Company {name: $c_name})
# # #                             MERGE (d:Document {file_name: $f_name})
# # #                             ON CREATE SET d.title = $f_name
# # #                             MERGE (c)-[:FILED]->(d)
# # #                             // Link generated entities (e.g., Risks, Concepts) back to the Company
# # #                             MATCH (e) 
# # #                             WHERE labels(e) IN ['Concept', 'Finding', 'Invoice', 'Buyer'] AND e.source_file IS NULL 
# # #                             MERGE (c)-[:RELATED_TO]->(e)
# # #                         """, f_name=file_name, c_name=company_name)

# # #                 except Exception as e:
# # #                     # Log failure but continue with vector search
# # #                     print(f"Warning: LLM Entity extraction failed: {e}")
            
# # #             return f"✅ Deleted {deleted_count} old nodes. Loaded {len(chunks)} chunks from {file_name} (Vector + Graph entities)."
            
# # #         except Exception as e:
# # #             return f"❌ Error processing PDF: {e}"
        
# # #         finally:
# # #             # Clean up temp file
# # #             if os.path.exists(temp_path): os.remove(temp_path)
    
# # #     return "❌ Unsupported file type."

# # # def remove_data(file_name):
# # #     """Removes all nodes and relationships related to a specific file"""
# # #     deleted_count = clear_data_by_source(file_name)
    
# # #     if deleted_count > 0:
# # #         return f"🗑️ Successfully deleted {deleted_count} nodes for {file_name}."
# # #     else:
# # #         return f"⚠️ No data found for {file_name}. Check the filename."


# # # Plan Final plus fixes
# # import os
# # import pandas as pd
# # from neo4j import GraphDatabase, exceptions
# # from langchain_community.document_loaders import PDFPlumberLoader
# # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# # from langchain_neo4j import Neo4jVector
# # from langchain_community.graphs import Neo4jGraph
# # from langchain_experimental.graph_transformers import LLMGraphTransformer

# # # --- 0. Setup Utilities ---
# # # These functions lazily initialize components using environment variables
# # # They rely on load_dotenv being run in app.py

# # def get_llm():
# #     """Lazy initialization of LLM"""
# #     LLM_MODEL = os.getenv("LLM_MODEL")
# #     GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
# #     return ChatGoogleGenerativeAI(model=LLM_MODEL, api_key=GOOGLE_KEY, temperature=0)

# # def get_embeddings():
# #     """Lazy initialization of embeddings"""
# #     EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
# #     GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
# #     return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=GOOGLE_KEY)

# # def get_neo4j_graph():
# #     """Lazy initialization of Neo4j graph"""
# #     NEO4J_URI = os.getenv("NEO4J_URI")
# #     NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# #     NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
# #     return Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# # def get_db_driver():
# #     """Returns a Neo4j Driver instance"""
# #     NEO4J_URI = os.getenv("NEO4J_URI")
# #     NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# #     NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
# #     return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# # def clear_data_by_source(file_name):
# #     """Deletes all nodes/relationships associated with a specific file_name"""
# #     driver = get_db_driver()
# #     delete_query = """
# #     MATCH (n) 
# #     WHERE n.source_file = $file_name OR n.file_name = $file_name
# #     DETACH DELETE n
# #     """
# #     with driver.session() as session:
# #         result = session.run(delete_query, file_name=file_name)
# #         deleted = result.consume().counters.nodes_deleted
# #         driver.close()
# #         return deleted

# # def ingest_data(uploaded_file, file_type, company_name):
# #     """Processes uploaded file and loads into Neo4j (Hybrid Structured + Vector)"""
# #     file_name = uploaded_file.name
    
# #     NEO4J_URI = os.getenv("NEO4J_URI")
# #     NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# #     NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    
# #     deleted_count = clear_data_by_source(file_name)

# #     # --- Excel/Structured Data Path (Tabular Data Modeling) ---
# #     if file_type == 'xlsx' or file_name.endswith('.csv'): # Handle both Excel and CSV uploads
        
# #         try:
# #             # Check for specific uploaded CSV names that were originally part of Excel
# #             if 'Inventory Register.csv' in file_name or 'MRI-Issued' in file_name:
# #                 df = pd.read_csv(uploaded_file, header=None)
# #                 sheet_name = 'Inventory Register'
# #             else:
# #                 # Assuming standard Excel upload
# #                 xlsx = pd.ExcelFile(uploaded_file)
# #                 sheet_name = 'Inventory Register'
# #                 df = xlsx.parse(sheet_name=sheet_name, header=None) # Read without header initially
            
# #             # --- CORRECTED INDEXING LOGIC ---
# #             # Assume row 0 and 1 contain metrics, row 2 contains the final meaningful headers
# #             # Master Metrics are derived from the first three rows
# #             master_and_header_df = df.iloc[0:3] 
            
# #             # The actual books inventory data starts from row 3 (index 3) 
# #             books_inventory_df = df.iloc[3:] 
            
# #             # The column headers are in the third row (index 2) of the raw Excel sheet
# #             books_inventory_df.columns = df.iloc[2] 
            
# #             # Clean up column headers and reset index
# #             books_inventory_df = books_inventory_df.rename(columns={books_inventory_df.columns[0]: 'Category'}) 
# #             books_inventory_df = books_inventory_df.reset_index(drop=True)
# #             books_inventory_df = books_inventory_df.dropna(subset=['Book Title']) # Drop rows without titles
            
# #         except Exception as e:
# #             # Return descriptive error for front-end
# #             return f"❌ Error reading Excel/CSV data structure. Details: {e}"

# #         def create_excel_metrics(tx):
# #             # 1. Create or MERGE Company node
# #             tx.run("MERGE (c:Company {name: $name})", name=company_name)
            
# #             # 2. Ingest MASTER Metrics (Totals from the top rows)
# #             # Accessing metrics that were assumed to be in the second column (index 1)
# #             master_metrics_data = {
# #                 'Total Opening Stock Amount': master_and_header_df.iloc[0, 1] if master_and_header_df.shape[1] > 1 else 0,
# #                 'Total Purchase Amount': master_and_header_df.iloc[1, 1] if master_and_header_df.shape[1] > 1 else 0,
# #                 'Total Sales Amount': master_and_header_df.iloc[2, 1] if master_and_header_df.shape[1] > 1 else 0,
# #             }
            
# #             for name, value in master_metrics_data.items():
# #                  tx.run("""
# #                     MATCH (c:Company {name: $c_name}) 
# #                     MERGE (m:Metric {name: $m_name, source_file: $f_name}) 
# #                     SET m.value = $m_value
# #                     MERGE (c)-[:HAS_METRIC]->(m)
# #                 """, c_name=company_name, m_name=name, m_value=value, f_name=file_name)

# #             # 3. Ingest BOOKS and INVOICES (Row-based entities)
# #             book_count = 0
# #             for idx, row in books_inventory_df.iterrows():
# #                 try:
# #                     book_title = str(row.get('Book Title', f"Book {idx}")).strip()
# #                     if book_title == 'nan' or not book_title: continue
                    
# #                     # Ensure units are treated as integers
# #                     op_u = pd.to_numeric(row.get('Opening Stock (Units)', 0), errors='coerce').fillna(0)
# #                     pur_u = pd.to_numeric(row.get('Purchase Units', 0), errors='coerce').fillna(0)

# #                     tx.run("""
# #                         MATCH (c:Company {name: $c_name})
# #                         MERGE (b:Book {title: $title, source_file: $f_name})
# #                         ON CREATE SET b.author = $author, b.isbn = $isbn, b.category = $cat, 
# #                                 b.opening_units = toInteger($op_u), b.purchase_units = toInteger($pur_u)
# #                         ON MATCH SET b.opening_units = toInteger($op_u), b.purchase_units = toInteger($pur_u)
# #                         MERGE (c)-[:SELLS]->(b)
# #                     """, c_name=company_name, title=book_title, f_name=file_name,
# #                         author=str(row.get('Author', 'Unknown')), isbn=str(row.get('ISBN', 'N/A')),
# #                         cat=str(row.get('Category', 'Unknown')),
# #                         op_u=op_u, pur_u=pur_u)

# #                     # Link to Store Location (if column exists)
# #                     if 'Store Location' in row and str(row['Store Location']).strip() != 'nan':
# #                         store_name = str(row['Store Location']).strip()
# #                         tx.run("""
# #                             MERGE (s:Store {name: $store_name})
# #                             MERGE (b:Book {title: $title, source_file: $f_name})
# #                             MERGE (b)-[:LOCATED_AT]->(s)
# #                         """, store_name=store_name, title=book_title, f_name=file_name)

# #                     book_count += 1
# #                 except Exception as e:
# #                     print(f"Skipping row {idx} due to processing error: {e}")
            
# #             # Create document metadata node
# #             tx.run("""
# #                 MERGE (d:Document {file_name: $f_name, type: 'Excel_Inventory'})
# #                 ON CREATE SET d.title = $title
# #                 WITH d
# #                 MATCH (c:Company {name: $c_name})
# #                 MERGE (c)-[:FILED]->(d)
# #             """, f_name=file_name, title=f"Inventory for {file_name}", c_name=company_name)
            
# #             return book_count

# #         driver = get_db_driver()
# #         try:
# #             with driver.session() as session:
# #                 book_count = session.execute_write(create_excel_metrics)
# #         finally:
# #             driver.close()
        
# #         return f"✅ Deleted {deleted_count} old nodes. Loaded {book_count} Books and Metrics from {file_name}."

# #     # --- PDF/Hybrid Data Path (Vector + Structured Graph) ---
# #     elif file_type == 'pdf':
        
# #         # Save temp file for processing
# #         temp_dir = "data_sources"
# #         os.makedirs(temp_dir, exist_ok=True)
# #         temp_path = os.path.join(temp_dir, file_name)
        
# #         with open(temp_path, "wb") as f:
# #             f.write(uploaded_file.getbuffer())
        
# #         try:
# #             loader = PDFPlumberLoader(temp_path)
# #             documents = loader.load()
            
# #             if not documents: return "❌ No content extracted from PDF."
            
# #             # Chunking for vector search
# #             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
# #             chunks = text_splitter.split_documents(documents)
            
# #             for chunk in chunks:
# #                 chunk.metadata['source_file'] = file_name 
# #                 chunk.metadata['company'] = company_name
            
# #             # 3. Vector Indexing (The Unstructured Part)
# #             embeddings_instance = get_embeddings()
# #             vector_db = Neo4jVector.from_documents(
# #                 chunks, embeddings_instance, url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
# #                 index_name="pdf_vector_index", node_label="Chunk"
# #             )
            
# #             # 4. LLM Graph Structuring (The Structured/Relational Part from PDF Text)
# #             # This is the "advanced" part for PDFs: extracting entities/relationships
# #             entity_extraction_enabled = True
            
# #             if entity_extraction_enabled and len(chunks) > 0:
# #                 try:
# #                     llm_instance = get_llm()
# #                     neo4j_graph_instance = get_neo4j_graph()
                    
# #                     # Define allowed node types for the LLM to extract from the PDF invoice
# #                     llm_transformer = LLMGraphTransformer(
# #                         llm=llm_instance,
# #                         allowed_nodes=['Company', 'Person', 'Product', 'Concept', 'Finding', 'Invoice', 'Buyer'],
# #                         allowed_relationships=['MENTIONS', 'HAS_RISK', 'DISCUSSES', 'ISSUED_TO', 'CONTAINS_ITEM']
# #                     )
                    
# #                     # Process first 5 chunks only for speed and cost control
# #                     sample_chunks = chunks[:5]
# #                     graph_documents = llm_transformer.convert_to_graph_documents(sample_chunks)
                    
# #                     if graph_documents:
# #                         neo4j_graph_instance.add_graph_documents(graph_documents)

# #                     # Link generated entities (e.g., Invoice, Buyer) to the Company/Document
# #                     driver = get_db_driver()
# #                     with driver.session() as session:
# #                          session.run("""
# #                             MERGE (c:Company {name: $c_name})
# #                             MERGE (d:Document {file_name: $f_name})
# #                             ON CREATE SET d.title = $f_name
# #                             MERGE (c)-[:FILED]->(d)
# #                             // Link generated entities (e.g., Invoice, Buyer) back to the Company
# #                             MATCH (e) 
# #                             WHERE labels(e) IN ['Concept', 'Finding', 'Invoice', 'Buyer'] AND e.source_file IS NULL 
# #                             MERGE (c)-[:RELATED_TO]->(e)
# #                         """, f_name=file_name, c_name=company_name)

# #                 except Exception as e:
# #                     # Log failure but continue with vector search
# #                     print(f"Warning: LLM Entity extraction failed: {e}")
            
# #             return f"✅ Deleted {deleted_count} old nodes. Loaded {len(chunks)} chunks from {file_name} (Vector + Graph entities)."
            
# #         except Exception as e:
# #             return f"❌ Error processing PDF: {e}"
        
# #         finally:
# #             # Clean up temp file
# #             if os.path.exists(temp_path): os.remove(temp_path)
    
# #     return "❌ Unsupported file type."

# # def remove_data(file_name):
# #     """Removes all nodes and relationships related to a specific file"""
# #     deleted_count = clear_data_by_source(file_name)
    
# #     if deleted_count > 0:
# #         return f"🗑️ Successfully deleted {deleted_count} nodes for {file_name}."
# #     else:
# #         return f"⚠️ No data found for {file_name}. Check the filename."

# # At last
# import os
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

# # Validate credentials
# # FIX: Corrected variable name from NEO4AJ_USERNAME to NEO4J_USERNAME
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
# - Use only the provided relationship types, node labels (e.g., Company, Metric, Book, Store, Invoice), and properties in the schema.
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
#             username=NEO4J_USERNAME,
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

#working
import os
import pandas as pd
from neo4j import GraphDatabase
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_neo4j import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
import re
from datetime import datetime

def get_llm():
    LLM_MODEL = os.getenv("LLM_MODEL")
    GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(model=LLM_MODEL, api_key=GOOGLE_KEY, temperature=0)

def get_embeddings():
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=GOOGLE_KEY)

def get_neo4j_graph():
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    return Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

def get_db_driver():
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def clear_data_by_source(file_name):
    """Deletes all nodes/relationships from a specific file"""
    driver = get_db_driver()
    delete_query = """
    MATCH (n) 
    WHERE n.source_file = $file_name OR n.file_name = $file_name
    DETACH DELETE n
    """
    with driver.session() as session:
        result = session.run(delete_query, file_name=file_name)
        deleted = result.consume().counters.nodes_deleted
        driver.close()
        return deleted

def clean_value(value):
    """Clean and convert values to appropriate types"""
    if pd.isna(value) or value == '' or str(value).strip() == '-':
        return None
    
    val_str = str(value).strip()
    
    # Try to parse as number
    try:
        if ',' in val_str:
            val_str = val_str.replace(',', '')
        return float(val_str)
    except:
        pass
    
    # Try to parse as date
    try:
        date_obj = pd.to_datetime(value)
        return date_obj.strftime('%Y-%m-%d')
    except:
        pass
    
    return val_str

def ingest_excel_intelligent(uploaded_file, company_name, file_name):
    """Intelligently ingests Excel - handles both simple and complex structures"""
    
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    except Exception as e:
        return f"❌ Error reading Excel: {e}"
    
    if df.empty:
        return "❌ Excel file is empty"
    
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    
    driver = get_db_driver()
    
    # Strategy: Detect structure type
    # If columns > 5 and rows > 1: treat as tabular data (each row = entity)
    # If columns <= 2: treat as key-value pairs (simple metrics)
    
    is_complex = len(df.columns) > 5 and len(df) > 1
    
    try:
        with driver.session() as session:
            # Create company node
            session.run("MERGE (c:Company {name: $name})", name=company_name)
            
            if is_complex:
                # COMPLEX TABULAR DATA - Each row is an entity
                return ingest_complex_excel(df, company_name, file_name, session)
            else:
                # SIMPLE KEY-VALUE - Traditional metrics
                return ingest_simple_excel(df, company_name, file_name, session)
    
    finally:
        driver.close()

def ingest_simple_excel(df, company_name, file_name, session):
    """Handles simple key-value Excel (Metric: Value)"""
    
    financial_metrics = {}
    
    for idx, row in df.iterrows():
        if len(row) >= 2:
            metric_name = str(row[0]).strip()
            metric_value = clean_value(row[1])
            
            if metric_name and metric_name.lower() not in ['nan', 'none', '']:
                financial_metrics[metric_name] = metric_value
    
    if not financial_metrics:
        return "❌ No valid metrics found"
    
    # Create metrics
    for name, value in financial_metrics.items():
        if value is not None:
            session.run("""
                MATCH (c:Company {name: $c_name}) 
                MERGE (m:Metric {name: $m_name, source_file: $f_name}) 
                SET m.value = $m_value, m.type = 'simple'
                MERGE (c)-[:HAS_METRIC]->(m)
            """, c_name=company_name, m_name=name, m_value=value, f_name=file_name)
    
    return f"✅ Loaded {len(financial_metrics)} metrics as key-value pairs."

def ingest_complex_excel(df, company_name, file_name, session):
    """Handles complex tabular Excel (inventory, transactions, etc.)"""
    
    columns = df.columns.tolist()
    entity_count = 0
    
    # Detect entity type from column names
    has_book_info = any('title' in str(col).lower() or 'book' in str(col).lower() or 'isbn' in str(col).lower() for col in columns)
    has_invoice_info = any('invoice' in str(col).lower() for col in columns)
    has_inventory_info = any('stock' in str(col).lower() or 'opening' in str(col).lower() for col in columns)
    
    # Determine primary entity label
    if has_book_info:
        entity_label = "Book"
    elif has_invoice_info:
        entity_label = "Invoice"
    elif has_inventory_info:
        entity_label = "InventoryItem"
    else:
        entity_label = "Record"
    
    for idx, row in df.iterrows():
        entity_id = f"{file_name}_{entity_label}_{idx}"
        
        # Build properties dictionary with SANITIZED column names
        properties = {
            'entity_id': entity_id,
            'source_file': file_name,
            'row_number': int(idx)
        }
        
        # Sanitize column names - remove special characters
        for col in columns:
            # Remove spaces, dots, parentheses, slashes, etc.
            col_name = str(col).strip()
            col_name = re.sub(r'[^\w]', '_', col_name)  # Replace non-alphanumeric with underscore
            col_name = re.sub(r'_+', '_', col_name)     # Replace multiple underscores with single
            col_name = col_name.strip('_')               # Remove leading/trailing underscores
            
            value = clean_value(row[col])
            if value is not None and col_name:
                properties[col_name] = value
        
        # Build SET clause dynamically
        set_clauses = [f"e.{k} = ${k}" for k in properties.keys()]
        set_string = ", ".join(set_clauses)
        
        # Create the entity node
        session.run(f"""
            MATCH (c:Company {{name: $c_name}})
            CREATE (e:{entity_label})
            SET {set_string}
            CREATE (c)-[:HAS_{entity_label.upper()}]->(e)
        """, c_name=company_name, **properties)
        
        entity_count += 1
        
        # Create additional structured nodes for key entities
        if has_book_info:
            create_book_relationships(session, row, entity_id, columns, company_name)
    
    # Create summary aggregate metrics
    create_aggregate_metrics(session, df, company_name, file_name, columns)
    
    return f"✅ Loaded {entity_count} {entity_label} records with full relational structure."

def create_book_relationships(session, row, entity_id, columns, company_name):
    """Creates related nodes for books (Author, Publisher, Store, etc.)"""
    
    # Author
    if 'Author' in columns:
        author = clean_value(row['Author'])
        if author:
            session.run("""
                MATCH (e {entity_id: $entity_id})
                MERGE (a:Author {name: $author})
                MERGE (e)-[:WRITTEN_BY]->(a)
            """, entity_id=entity_id, author=author)
    
    # Publisher
    if 'Publisher' in columns:
        publisher = clean_value(row['Publisher'])
        if publisher:
            session.run("""
                MATCH (e {entity_id: $entity_id})
                MERGE (p:Publisher {name: $publisher})
                MERGE (e)-[:PUBLISHED_BY]->(p)
            """, entity_id=entity_id, publisher=publisher)
    
    # Store Location
    if 'Store Location' in columns:
        store = clean_value(row['Store Location'])
        if store:
            session.run("""
                MATCH (e {entity_id: $entity_id})
                MERGE (s:Store {location: $store})
                MERGE (e)-[:STORED_AT]->(s)
            """, entity_id=entity_id, store=store)

def create_aggregate_metrics(session, df, company_name, file_name, columns):
    """Creates summary metrics from tabular data"""
    
    numeric_columns = df.select_dtypes(include=['number']).columns
    
    for col in numeric_columns:
        col_clean = str(col).strip()
        
        # Total
        total = df[col].sum()
        if not pd.isna(total):
            session.run("""
                MATCH (c:Company {name: $c_name})
                MERGE (m:Metric {name: $m_name, source_file: $f_name})
                SET m.value = $m_value, m.type = 'aggregate'
                MERGE (c)-[:HAS_METRIC]->(m)
            """, c_name=company_name, m_name=f"Total {col_clean}", m_value=float(total), f_name=file_name)
        
        # Average
        avg = df[col].mean()
        if not pd.isna(avg):
            session.run("""
                MATCH (c:Company {name: $c_name})
                MERGE (m:Metric {name: $m_name, source_file: $f_name})
                SET m.value = $m_value, m.type = 'aggregate'
                MERGE (c)-[:HAS_METRIC]->(m)
            """, c_name=company_name, m_name=f"Average {col_clean}", m_value=float(avg), f_name=file_name)

def ingest_pdf_intelligent(uploaded_file, company_name, file_name):
    """Intelligently ingests PDF - extracts both structure and text"""
    
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    
    temp_dir = "data_sources"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file_name)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        loader = PDFPlumberLoader(temp_path)
        documents = loader.load()
        
        if not documents:
            return "❌ No content extracted from PDF"
        
        full_text = "\n\n".join([doc.page_content for doc in documents])
        
        # Extract structured data from invoice-like PDFs
        invoice_data = extract_invoice_data(full_text, file_name)
        
        if invoice_data:
            driver = get_db_driver()
            try:
                with driver.session() as session:
                    create_invoice_nodes(session, invoice_data, company_name, file_name)
            finally:
                driver.close()
        
        # Create vector embeddings for semantic search
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        for chunk in chunks:
            chunk.metadata['source_file'] = file_name
            chunk.metadata['company'] = company_name
        
        embeddings = get_embeddings()
        
        vector_db = Neo4jVector.from_documents(
            chunks,
            embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name="pdf_vector_index",
            node_label="Chunk",
            text_node_property="text",
            embedding_node_property="embedding",
            retrieval_query="""
                RETURN node.text AS text, score, 
                {source: node.source_file, type: "PDF Chunk"} AS metadata
            """
        )
        
        structure_msg = f" + {len(invoice_data)} structured entities" if invoice_data else ""
        return f"✅ Loaded {len(chunks)} text chunks{structure_msg} from {file_name}."
    
    except Exception as e:
        return f"❌ Error processing PDF: {e}"
    
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

def extract_invoice_data(text, file_name):
    """Extracts structured data from invoice text"""
    
    invoice_data = {}
    
    # Invoice number
    inv_match = re.search(r'Invoice No[.:]?\s*([A-Z0-9-]+)', text, re.IGNORECASE)
    if inv_match:
        invoice_data['invoice_no'] = inv_match.group(1)
    
    # Date
    date_match = re.search(r'Invoice Date[:]?\s*(\d{2}-\w{3}-\d{4})', text, re.IGNORECASE)
    if date_match:
        invoice_data['invoice_date'] = date_match.group(1)
    
    # Customer
    buyer_match = re.search(r'Buyer \(Bill To\)\s*([^\n]+)', text, re.IGNORECASE)
    if buyer_match:
        invoice_data['customer'] = buyer_match.group(1).strip()
    
    # Total
    total_match = re.search(r'Total[:\s]+[₹n]+\s*([\d,]+\.?\d*)', text, re.IGNORECASE)
    if total_match:
        invoice_data['total_amount'] = float(total_match.group(1).replace(',', ''))
    
    # GST
    gst_match = re.search(r'GST @\d+%[:\s]+[₹n]+\s*([\d,]+\.?\d*)', text, re.IGNORECASE)
    if gst_match:
        invoice_data['gst_amount'] = float(gst_match.group(1).replace(',', ''))
    
    # Item description
    item_match = re.search(r'Particulars.*?1\s+([^\n]+)', text, re.IGNORECASE | re.DOTALL)
    if item_match:
        invoice_data['item_description'] = item_match.group(1).strip()[:100]
    
    invoice_data['source_file'] = file_name
    
    return invoice_data if len(invoice_data) > 2 else None

def create_invoice_nodes(session, invoice_data, company_name, file_name):
    """Creates structured invoice nodes"""
    
    session.run("""
        MATCH (c:Company {name: $c_name})
        CREATE (i:Invoice)
        SET i += $props
        CREATE (c)-[:ISSUED_INVOICE]->(i)
    """, c_name=company_name, props=invoice_data)
    
    # Create customer node if present
    if 'customer' in invoice_data:
        session.run("""
            MATCH (i:Invoice {invoice_no: $inv_no})
            MERGE (cust:Customer {name: $cust_name})
            MERGE (i)-[:BILLED_TO]->(cust)
        """, inv_no=invoice_data.get('invoice_no'), cust_name=invoice_data['customer'])

def ingest_data(uploaded_file, file_type, company_name):
    """Main ingestion router"""
    
    file_name = uploaded_file.name
    deleted_count = clear_data_by_source(file_name)
    
    if file_type == 'xlsx':
        result = ingest_excel_intelligent(uploaded_file, company_name, file_name)
        return f"🗑️ Deleted {deleted_count} old nodes.\n{result}"
    
    elif file_type == 'pdf':
        result = ingest_pdf_intelligent(uploaded_file, company_name, file_name)
        return f"🗑️ Deleted {deleted_count} old nodes.\n{result}"
    
    return "❌ Unsupported file type"

def remove_data(file_name):
    """Removes all data for a file"""
    deleted_count = clear_data_by_source(file_name)
    
    if deleted_count > 0:
        return f"🗑️ Deleted {deleted_count} nodes for {file_name}."
    else:
        return f"⚠️ No data found for {file_name}."