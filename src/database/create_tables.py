import psycopg2
from psycopg2 import sql

conn = psycopg2.connect(database="capstone2",
                        user="postgres",
                        host="localhost", port="5435")
cur = conn.cursor()

# Create tables
cur.execute("""
    CREATE TABLE papers
    (identifier TEXT,
    url TEXT,
    title TEXT,
    set_spec TEXT,
    subjects TEXT,
    authors TEXT,
    dates TEXT,
    description TEXT);
""")
conn.commit()