import click

import psycopg2
from psycopg2 import sql

conn = psycopg2.connect(database="capstone2",
                        user="postgres",
                        host="localhost", port="5435")
cur = conn.cursor()

@click.group()
def cli():
    pass

@cli.command()
@click.option('--name', default=None, required=True, type=str, help='Name of the table to delete.')
def drop_table(name):
    """Note: This function is vulnerable to SQL injection and should only be used for testing purposes locally."""
    try:
        cur.execute(f"DROP TABLE {name};")
        conn.commit()
        print(f'deleted {name}')
    except psycopg2.errors.UndefinedTable:
        print(f"table {name} doesn't exist")

if __name__ == "__main__":
    cli()
