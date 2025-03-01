import os
import sys
import sqlite3
from sqlalchemy import inspect

# Add the parent directory to the path so we can import the app modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import engine, Base
from app.models import User, Trip, StatusUpdate, Location, Issue, Notification

def create_tables():
    """Create all tables in the database."""
    # First, ensure the database file exists and is writable
    db_path = "./app.db"
    
    # Remove the existing database file if it exists
    if os.path.exists(db_path):
        print(f"Removing existing database file: {db_path}")
        os.remove(db_path)
    
    # Create a new database file
    print(f"Creating new database file: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.close()
    
    # Check if the file was created
    if os.path.exists(db_path):
        print(f"Database file created: {db_path}")
        print(f"File size: {os.path.getsize(db_path)} bytes")
    else:
        print(f"ERROR: Failed to create database file: {db_path}")
        return
    
    # Create tables using SQLAlchemy
    print("Creating tables using SQLAlchemy...")
    Base.metadata.create_all(bind=engine)
    
    # Check if tables were created
    inspector = inspect(engine)
    new_tables = inspector.get_table_names()
    print(f"Tables after creation: {new_tables}")
    
    # Verify specific tables
    for table in ['users', 'trips', 'status_updates', 'locations', 'issues', 'notifications']:
        if table in new_tables:
            print(f"Table '{table}' created successfully.")
        else:
            print(f"ERROR: Table '{table}' was not created!")

if __name__ == "__main__":
    create_tables()
    print("Database initialization complete.") 