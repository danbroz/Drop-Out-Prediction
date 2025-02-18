# Import necessary libraries
import numpy as np                  # For numerical operations
import pandas as pd                 # For data manipulation
from cassandra.cluster import Cluster  # To connect and query the Cassandra database
from sklearn.linear_model import LogisticRegression  # For the Logistic Regression classifier
from sklearn.model_selection import train_test_split  # For splitting the data (optional)

def fetch_data_from_cassandra():
    """
    Connect to the Cassandra database and fetch 70 rows of student data.
    
    Expected columns in the 'students' table:
        - attendance: Numeric value indicating attendance rate or days attended.
        - prev_gpa: The student's previous year's GPA.
        - dropout: Binary indicator (1 means the student dropped out, 0 means they did not).
    
    Returns:
        A pandas DataFrame containing the fetched data.
    """
    # Connect to the Cassandra cluster.
    # Replace '127.0.0.1' with your Cassandra node's IP address if necessary.
    cluster = Cluster(['127.0.0.1'])
    
    # Connect to the keyspace named 'school_data'.
    session = cluster.connect('school_data')
    
    # Define the CQL query to select the necessary columns from the 'students' table.
    # LIMIT 70 ensures we only retrieve 70 rows.
    query = "SELECT attendance, prev_gpa, dropout FROM students LIMIT 70"
    rows = session.execute(query)
    
    # Convert the Cassandra rows into a list of dictionaries.
    data = []
    for row in rows:
        data.append({
            'attendance': row.attendance,
            'prev_gpa': row.prev_gpa,
            'dropout': row.dropout
        })
    
    # Shutdown the session and cluster connection once the data is fetched.
    session.shutdown()
    cluster.shutdown()
    
    # Create a pandas DataFrame from the data list for easier manipulation.
    df = pd.DataFrame(data)
    return df

def train_logistic_regression_classifier(df):
    """
    Train a Logistic Regression classifier using student data.
    
    The classifier uses the following features:
        - attendance
        - prev_gpa
    to predict the dropout status (0 or 1).
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing the columns 'attendance', 'prev_gpa', and 'dropout'.
    
    Returns:
        model (LogisticRegression): The trained Logistic Regression classifier.
    """
    # Extract features (attendance and previous GPA) as a NumPy array.
    X = df[['attendance', 'prev_gpa']].values
    
    # Extract the target dropout labels.
    y = df['dropout'].values
    
    # OPTIONAL: You can split the data into training and testing sets.
    # For this example with a small dataset, we will train on the entire dataset.
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create the Logistic Regression classifier.
    # Setting random_state ensures reproducibility.
    model = LogisticRegression(random_state=42, solver='liblinear')
    
    # Train the classifier on the entire dataset.
    model.fit(X, y)
    
    # OPTIONAL: Evaluate the model's performance on the training data.
    train_accuracy = model.score(X, y)
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    
    # Return the trained classifier.
    return model

def predict_dropout(model, new_data):
    """
    Predict the dropout status for new student data using the trained Logistic Regression classifier.
    
    Parameters:
        model (LogisticRegression): The trained Logistic Regression model.
        new_data (list or array): New data points where each point is a list of two features: [attendance, prev_gpa].
    
    Returns:
        predictions (list): Predicted dropout statuses (0 or 1) for each new data point.
    """
    # Convert new_data into a NumPy array for consistency.
    new_data = np.array(new_data)
    
    # Use the Logistic Regression model to predict dropout status for the new data.
    predictions = model.predict(new_data)
    
    # Return the list of predictions.
    return predictions

# Main execution starts here.
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Step 1: Fetch data from the Cassandra database.
    # -------------------------------------------------------------------------
    print("Fetching data from Cassandra database...")
    df = fetch_data_from_cassandra()
    print("Data fetched successfully!")
    print("Sample data:")
    print(df.head())  # Display the first few rows to verify the data.
    
    # -------------------------------------------------------------------------
    # Step 2: Train the Logistic Regression classifier using the fetched data.
    # -------------------------------------------------------------------------
    print("\nTraining Logistic Regression classifier...")
    logistic_model = train_logistic_regression_classifier(df)
    print("Training complete!")
    
    # -------------------------------------------------------------------------
    # Step 3: Use the trained model to predict dropout status for new students.
    # -------------------------------------------------------------------------
    # Example new student data:
    # Each sub-list represents [attendance, prev_gpa].
    new_students = [
        [90, 3.5],  # Student with high attendance and a strong GPA.
        [60, 2.0],  # Student with lower attendance and a lower GPA.
        [80, 3.0]   # Student with moderate attendance and GPA.
    ]
    
    print("\nPredicting dropout status for new student data...")
    dropout_predictions = predict_dropout(logistic_model, new_students)
    
    # Display the predictions in a user-friendly format.
    for i, prediction in enumerate(dropout_predictions):
        status = "Dropout" if prediction == 1 else "No Dropout"
        print(f"Student {i+1}: Predicted status - {status}")

