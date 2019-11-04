import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT # <-- ADD THIS LINE

conn = psycopg2.connect(database="capstone2",
                        user="postgres",
                        host="localhost", port="5435")
cur = conn.cursor()

# Create tables
cur.execute("CREATE TABLE test (id serial PRIMARY KEY, num integer, data varchar);")
conn.commit()
