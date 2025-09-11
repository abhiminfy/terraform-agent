import os
import sqlite3

DB_PATH = os.getenv("CHAT_DB_PATH", "agent.db")
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute(
    """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT,
    role TEXT,
    content TEXT,
    metadata TEXT,
    timestamp TEXT
)
"""
)
conn.commit()
conn.close()
