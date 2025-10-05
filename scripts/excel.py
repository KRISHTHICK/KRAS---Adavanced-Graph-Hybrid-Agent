import pandas as pd
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

# --- DB Connection Setup ---
URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

# The Company we are focusing on
COMPANY_NAME = "Zenalyst Corp." 

def load_excel_data(tx):
    # 1. Read a mock data point from the Excel file
    # HACKATHON SIMPLIFICATION: Assume you know the file path and relevant data points
    # In a real scenario, you'd use pandas to search the Excel content dynamically.

    # --- Create Company and Document Nodes ---
    tx.run("CREATE (c:Company {name: $name, source: 'Excel'})", name=COMPANY_NAME)
    tx.run("CREATE (d:Document {title: '2024 Financial Report', source: 'Financial_Metrics.xlsx'})")
    tx.run("MATCH (c:Company), (d:Document) WHERE c.name = $name "
           "CREATE (c)-[:FILED]->(d)", name=COMPANY_NAME)

    # --- Create Metrics (Replace with real data extracted via pandas) ---
    financial_metrics = {
        "Equity": 1200000,
        "Debt": 500000,
        "TaxRate": 0.21
    }

    for name, value in financial_metrics.items():
        tx.run("MATCH (c:Company {name: $c_name}) "
               "CREATE (m:Metric {name: $m_name, value: $m_value, source_file: 'Financial_Metrics.xlsx'}) "
               "CREATE (c)-[:HAS_METRIC]->(m)", 
               c_name=COMPANY_NAME, m_name=name, m_value=value)

    print(f"✅ Successfully loaded {len(financial_metrics)} metrics and core structure for {COMPANY_NAME}.")


# --- Main Execution ---
if __name__ == "__main__":
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    with driver.session() as session:
        # OPTIONAL: Clear existing data for a clean run (ONLY DO THIS IN TEST ENV)
        session.run("MATCH (n) DETACH DELETE n") 

        session.execute_write(load_excel_data)
    driver.close()