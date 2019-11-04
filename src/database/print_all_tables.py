import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT # <-- ADD THIS LINE

conn = psycopg2.connect(database="capstone2",
                        user="postgres",
                        host="localhost", port="5435")
cur = conn.cursor()

cur.execute("""SELECT table_name FROM information_schema.tables 
       WHERE table_schema = 'public'""") 
for table in cur.fetchall(): 
    print(table)
