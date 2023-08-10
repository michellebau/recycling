import sqlite3, csv

# Connect to a new SQLite database
conn = sqlite3.connect('recycling_info.db')
cursor = conn.cursor()

# Create the Items table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Items (
    item_id INTEGER PRIMARY KEY,
    item_name TEXT NOT NULL,
    description TEXT
)
''')

# Create the RecyclingSteps table
cursor.execute('''
CREATE TABLE IF NOT EXISTS RecyclingSteps (
    step_id INTEGER PRIMARY KEY,
    item_id INTEGER,
    step_description TEXT NOT NULL,
    FOREIGN KEY (item_id) REFERENCES Items(item_id)
)
''')

# Import data from CSV files

# For the Items table
with open('items.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        cursor.execute("INSERT INTO Items(item_id, item_name, description) VALUES (?, ?, ?)",
                       (row['item_id'], row['item_name'], row['description']))

# For the RecyclingSteps table
with open('recycling_steps.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        cursor.execute("INSERT INTO RecyclingSteps(step_id, item_id, step_description) VALUES (?, ?, ?)",
                       (row['step_id'], row['item_id'], row['step_description']))


# Commit the changes and close the connection
conn.commit()
conn.close()
