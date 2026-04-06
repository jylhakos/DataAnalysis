"""
Database setup script for PostgreSQL with pgvector.
This script creates the necessary tables and indexes if they don't exist.
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sys

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'vectordb',
    'user': 'postgres',
    'password': 'postgres'
}


def setup_database():
    """Set up the database with pgvector extension and tables."""
    try:
        # Connect to PostgreSQL
        print("Connecting to PostgreSQL...")
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # Enable pgvector extension
        print("Enabling pgvector extension...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Verify extension
        cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        result = cur.fetchone()
        if result:
            print(f"✓ pgvector extension enabled: {result[0]}")
        else:
            print("✗ Failed to enable pgvector extension")
            sys.exit(1)

        # Check if table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE  table_schema = 'public'
                AND    table_name   = 'documents'
            );
        """)
        table_exists = cur.fetchone()[0]

        if table_exists:
            print("✓ Documents table already exists")
            # Get row count
            cur.execute("SELECT COUNT(*) FROM documents;")
            count = cur.fetchone()[0]
            print(f"  Current document count: {count}")
        else:
            print("Table does not exist (will be created by init.sql)")

        # List all tables
        cur.execute("""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public';
        """)
        tables = cur.fetchall()
        print(f"\nAvailable tables: {[t[0] for t in tables]}")

        # List all indexes
        cur.execute("""
            SELECT indexname FROM pg_indexes 
            WHERE tablename = 'documents';
        """)
        indexes = cur.fetchall()
        if indexes:
            print(f"Indexes on documents table: {[i[0] for i in indexes]}")

        cur.close()
        conn.close()

        print("\n✓ Database setup complete!")
        return True

    except psycopg2.Error as e:
        print(f"\n✗ Database error: {e}")
        print("\nMake sure PostgreSQL is running:")
        print("  docker compose up -d")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Database Setup Script")
    print("=" * 60)
    success = setup_database()
    sys.exit(0 if success else 1)
