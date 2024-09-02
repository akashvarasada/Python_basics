import sqlite3

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('Desktop App\Database\python_mwf_10.db')

# Create a cursor object to execute SQL commands
cursor = conn.cursor()

# Create the 'student' table
cursor.execute('''
CREATE TABLE student (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Fname TEXT,
    Lname TEXT,
    Email TEXT,
    Mobile BIG INTEGER
)
''')

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Database and table created successfully.")
