import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT # <-- ADD THIS LINE

conn = psycopg2.connect(dbname='postgres',
      user='postgres', host='localhost', port='5435')

conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT) # <-- ADD THIS LINE

cur = conn.cursor()

# Create database
cur.execute(
    sql.SQL("CREATE DATABASE {}").format(sql.Identifier('capstone2'))
    )
