import os
import dotenv

dotenv.load_dotenv()

connection_args = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD").replace("@", "%40") if os.getenv("DB_PASSWORD") else "",
    "db_name": os.getenv("DB_NAME"),
}