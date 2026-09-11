import pandas as pd
from sqlalchemy import create_engine

# Load CSV
df = pd.read_csv('data/raw_data/telco_customers.csv')

# PASTE YOUR CONNECTION STRING HERE (from Supabase Direct tab)
connection_string = "postgresql://postgres:christtheredeeme@db.jfdujgptoaxmdboepwua.supabase.co:5432/postgres"

# Connect to database
engine = create_engine(connection_string)

# Load data into Supabase
df.to_sql('telco_customers', engine, if_exists='replace', index=False)

print(f"✅ Loaded {len(df)} rows into telco_customers table")