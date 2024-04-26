import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()


def init_db():
    HOST = os.getenv('HOST')
    USER = os.getenv('USER')
    PASSWORD = os.getenv('PASSWORD')

    conn = psycopg2.connect(
                        host=HOST,
                        user=USER,
                        password=PASSWORD
                        )
    return conn