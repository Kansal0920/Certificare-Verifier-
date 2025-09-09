import sqlite3

def init_db():
    conn = sqlite3.connect("sample.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS certificates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        roll_no TEXT,
        name TEXT,
        certificate_id TEXT,
        marks TEXT,
        blockchain_hash TEXT
    )
    """)

    # Insert dummy records
    sample_data = [
        ("2023/368", "Kulpreet Singh", "CERT123", "85%", "0xabc123"),
        ("2023/390", "Navan Kaur", "CERT456", "90%", "0xdef456"),
        ("2023/349", "Japjeet Singh", "CERT789", "88%", "0xghi789")
    ]

    cursor.executemany("INSERT INTO certificates (roll_no, name, certificate_id, marks, blockchain_hash) VALUES (?, ?, ?, ?, ?)", sample_data)
    conn.commit()
    conn.close()

def get_connection():
    return sqlite3.connect("sample.db")
