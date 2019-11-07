import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

conn = psycopg2.connect(dbname='postgres',
    user='postgres', host='localhost', port='5435')

conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

cur = conn.cursor()

# Create database
try:
    cur.execute(
        sql.SQL("CREATE DATABASE {}").format(sql.Identifier('capstone2'))
    )
except psycopg2.errors.DuplicateDatabase:
    print('database already exists')
