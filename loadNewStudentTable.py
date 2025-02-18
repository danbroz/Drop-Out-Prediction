import pandas as pd
import uuid
from cassandra.cluster import Cluster

def load_csv_to_cassandra(csv_file):
    """
    Reads student data from a CSV file and inserts the data into a new Cassandra table called newStudents.
    
    Expected CSV columns (adjust as needed):
        - attendance: Numeric value (e.g., attendance rate or days attended)
        - prev_gpa: Numeric value representing the previous year's GPA
        - dropout: Binary or integer value indicating dropout status (optional)
    
    The function connects to a Cassandra keyspace called 'school_data' and creates the newStudents table if it does not exist.
    """
    # Step 1: Read CSV file into a DataFrame.
    df = pd.read_csv(csv_file)
    print("CSV data loaded successfully:")
    print(df.head())
    
    # Step 2: Connect to the Cassandra cluster.
    # Replace '127.0.0.1' with your Cassandra node's IP address if necessary.
    cluster = Cluster(['127.0.0.1'])
    session = cluster.connect('school_data')
    
    # Step 3: Create the newStudents table if it doesn't exist.
    # Adjust the table schema as necessary. Here we use a UUID as primary key.
    create_table_query = """
    CREATE TABLE IF NOT EXISTS newStudents (
        student_id uuid PRIMARY KEY,
        attendance float,
        prev_gpa float,
        dropout int
    );
    """
    session.execute(create_table_query)
    print("Ensured 'newStudents' table exists.")
    
    # Step 4: Insert each row from the DataFrame into the newStudents table.
    insert_query = """
    INSERT INTO newStudents (student_id, attendance, prev_gpa, dropout)
    VALUES (%s, %s, %s, %s)
    """
    
    for index, row in df.iterrows():
        student_id = uuid.uuid4()
        # Retrieve values from the DataFrame row; adjust column names if necessary.
        attendance = row['attendance']
        prev_gpa = row['prev_gpa']
        # If dropout column is not present in your CSV, you may use a default value (e.g., 0)
        dropout = row.get('dropout', 0) if 'dropout' in row else 0
        
        session.execute(insert_query, (student_id, attendance, prev_gpa, dropout))
    
    print("All records have been transferred to the Cassandra table 'newStudents'.")
    
    # Step 5: Close the session and cluster connection.
    session.shutdown()
    cluster.shutdown()

if __name__ == "__main__":
    # Specify the path to your CSV file
    csv_file = "student_data.csv"
    load_csv_to_cassandra(csv_file)

