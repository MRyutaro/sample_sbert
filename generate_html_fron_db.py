import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database connection settings
DB_ENGINE = os.getenv("LOCAL_DB_ENGINE")
DB_NAME = os.getenv("LOCAL_DB_NAME")
DB_USER = os.getenv("LOCAL_DB_USER")
DB_PASSWORD = os.getenv("LOCAL_DB_PASSWORD")
DB_HOST = os.getenv("LOCAL_DB_HOST")
DB_PORT = os.getenv("LOCAL_DB_PORT")

# Connect to the database
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)

# Create a cursor to interact with the database
cur = conn.cursor()

# Query to get article information
query = "SELECT id, title, body FROM articles"
cur.execute(query)

# Fetch all articles
articles = cur.fetchall()

# Generate HTML files for each article
for article in articles:
    article_id, title, body = article
    file_name = f".\\data\\{article_id}.html"

    html_content = body

    # Save the HTML content to a file
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(html_content)

# Close the cursor and connection
cur.close()
conn.close()
