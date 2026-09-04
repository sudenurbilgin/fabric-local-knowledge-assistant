import sqlite3
import json


def print_document(row):
    document_id, content, embedding_text = row
    embedding = json.loads(embedding_text)
    print(f"ID: {document_id}")
    print(f"Content: {content}")
    print(f"Embedding: {embedding}")


def main():
    connection = sqlite3.connect("sqlite_sandbox.db")
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )

    cursor.execute("DELETE FROM documents")

    records = [
        (1, "Foundry Local runs AI models directly on your device.", [0.12, -0.08, 0.44]),
        (2, "Embedding models convert text into numerical vectors.", [0.71, 0.10, -0.22]),
        (3, "SQLite is a lightweight local database.", [-0.16, 0.62, 0.31]),
    ]

    cursor.executemany(
        "INSERT INTO documents (id, content, embedding) VALUES (?, ?, ?)",
        [
            (document_id, content, json.dumps(embedding))
            for document_id, content, embedding in records
        ],
    )
    connection.commit()

    print("All documents:")
    cursor.execute("SELECT id, content, embedding FROM documents ORDER BY id")
    for row in cursor.fetchall():
        print_document(row)
        print()

    print("Document with id=2:")
    cursor.execute(
        "SELECT id, content, embedding FROM documents WHERE id = ?",
        (2,),
    )
    print_document(cursor.fetchone())
    print()

    print("Search for 'SQLite':")
    cursor.execute(
        "SELECT id, content, embedding FROM documents WHERE content LIKE ?",
        ("%SQLite%",),
    )
    print_document(cursor.fetchone())

    connection.close()

    connection = sqlite3.connect("sqlite_sandbox.db")
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM documents")
    row_count = cursor.fetchone()[0]
    print(f"\nPersistent row count after reopening database: {row_count}")
    connection.close()


if __name__ == "__main__":
    main()
