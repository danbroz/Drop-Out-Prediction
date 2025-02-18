# Import necessary libraries
import pandas as pd              # For reading and manipulating CSV data
import uuid                      # To generate unique IDs for each student record
from cassandra.cluster import Cluster  # To connect and interact with the Cassandra database

def transfer_csv_to_cassandra(csv_file):
    """
    Reads student data from a CSV file and inserts the data into a Cassandra table.
    
    Expected CSV columns:
        - attendance: Numeric value (e.g., attendance rate or days attended)
        - prev_gpa: Numeric value representing the previous year's GPA
        - dropout: Binary or integer value indicating dropout status
    
    The function connects to a Cassandra keyspace named 'school_data' and inserts each record
    into the 'students' table. The table is created if it does not already exist.
    
    Parameters:
        csv_file (str): Path to the CSV file containing student data.
    """
    # Step 1: Read CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    print("CSV data loaded successfully:")
    print(df.head())

    # Step 2: Connect to the Cassandra cluster
    # Replace '127.0.0.1' with your Cassandra node's IP address if needed.
    cluster = Cluster(['127.0.0.1'])
    # Connect to the keyspace 'school_data'
    session = cluster.connect('school_data')
    
    # Step 3 (Optional): Create the 'students' table if it doesn't exist.
    # This table has a UUID primary key and columns for attendance, prev_gpa, and dropout.
    create_table_query = """
    CREATE TABLE IF NOT EXISTS students (
        student_id uuid PRIMARY KEY,
        attendance float,
        prev_gpa float,
        dropout int
    );
    """
    session.execute(create_table_query)
    print("Ensured 'students' table exists.")

    # Step 4: Insert each row from the DataFrame into the Cassandra table.
    # We'll generate a unique UUID for each student.
    insert_query = """
    INSERT INTO students (student_id, attendance, prev_gpa, dropout)
    VALUES (%s, %s, %s, %s)
    """
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Generate a unique UUID for this student record
        student_id = uuid.uuid4()
        # Retrieve values from the DataFrame row
        attendance = row['attendance']
        prev_gpa = row['prev_gpa']
        dropout = row['dropout']
        
        # Execute the insert statement with the row's data
        session.execute(insert_query, (student_id, attendance, prev_gpa, dropout))
    
    print("All records have been transferred to the Cassandra database.")

    # Step 5: Shutdown the session and cluster connection
    session.shutdown()
    cluster.shutdown()

# Main execution starts here.
if __name__ == "__main__":
    # Specify the path to your CSV file
    csv_file = "student_data.csv"
    transfer_csv_to_cassandra(csv_file)

