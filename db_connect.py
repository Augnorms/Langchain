import mysql.connector
from dotenv import load_dotenv
import os

load_dotenv()



def database_connect():
    return mysql.connector.connect(
        host=os.getenv("DBHOST"),
        user=os.getenv("DBUSER"),
        password=os.getenv("MYPASSWORD"),
        database=os.getenv("DBNAME")
    )


def save_message_to_db(role, message):
    if(database_connect):
        conn = database_connect()
        cursor = conn.cursor()
        sql = "INSERT INTO chat_history (role, message) VALUES (%s, %s)"
        cursor.execute(sql, (role, message))
        conn.commit()
        cursor.close()
        conn.close()
    else:
        print("database connection failed")