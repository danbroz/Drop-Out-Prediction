# Import necessary libraries
import numpy as np                  # For numerical operations
import pandas as pd                 # For data manipulation
from cassandra.cluster import Cluster  # To connect and query the Cassandra database
from sklearn.neighbors import KNeighborsClassifier  # To perform K-Nearest Neighbors classification
from sklearn.model_selection import train_test_split   # To split the data for training/testing

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
    # Replace '127.0.0.1' with the IP(s) of your Cassandra node(s) if necessary.
    cluster = Cluster(['127.0.0.1'])
    # Connect to the keyspace named 'school_data'.
    session = cluster.connect('school_data')
    
    # Define a CQL query to select the necessary columns from the 'students' table.
    # LIMIT 70 ensures we only get 70 rows.
    query = "SELECT attendance, prev_gpa, dropout FROM students LIMIT 70"
    rows = session.execute(query)
    
    # Convert the result rows into a list of dictionaries.
    data = []
    for row in rows:
        data.append({
            'attendance': row.attendance,
            'prev_gpa': row.prev_gpa,
            'dropout': row.dropout
        })
    
    # Shutdown the session and cluster connection once done.
    session.shutdown()
    cluster.shutdown()
    
    # Create a pandas DataFrame from the list for easier data manipulation.
    df = pd.DataFrame(data)
    return df

def train_knn_classifier(df, n_neighbors=3):
    """
    Train a K-Nearest Neighbors classifier using student data.
    The classifier will use the following features:
        - attendance
        - prev_gpa
    to predict the dropout status (0 or 1).

    Parameters:
        df (pandas.DataFrame): DataFrame containing the columns 'attendance', 'prev_gpa', and 'dropout'.
        n_neighbors (int): Number of neighbors to use for the KNN classifier (default is 3).

    Returns:
        knn (KNeighborsClassifier): The trained KNeighborsClassifier model.
    """
    # Extract features from the DataFrame (attendance and previous GPA).
    X = df[['attendance', 'prev_gpa']].values  # Convert to a NumPy array for modeling.
    
    # Extract the target dropout labels.
    y = df['dropout'].values
    
    # OPTIONAL: If you wish to evaluate the classifier on a hold-out set, you can split the data.
    # Here, we train on the entire dataset since it's small (70 data points).
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # For simplicity, we use the full dataset for training.
    
    # Create the K-Nearest Neighbors model.
    # The parameter n_neighbors specifies how many neighbors to consider.
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Train the KNN model using the feature matrix X and target vector y.
    knn.fit(X, y)
    
    # OPTIONAL: Evaluate the model's performance on the training data.
    # This is only for demonstration since training accuracy is not a reliable metric.
    train_accuracy = knn.score(X, y)
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    
    # Return the trained KNN model.
    return knn

def predict_dropout(knn, new_data):
    """
    Predict the dropout status for new student data using the trained KNN classifier.

    Parameters:
        knn (KNeighborsClassifier): The trained KNN model.
        new_data (list or array): New data points where each point is a list of two features: [attendance, prev_gpa].

    Returns:
        predictions (list): Predicted dropout statuses (0 or 1) for each new data point.
    """
    # Convert new_data into a NumPy array in case it isn't one already.
    new_data = np.array(new_data)
    
    # Use the KNN model to predict dropout statuses for the new data points.
    predictions = knn.predict(new_data)
    
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
    # Step 2: Train the KNN classifier using the fetched data.
    # -------------------------------------------------------------------------
    print("\nTraining K-Nearest Neighbors classifier...")
    # You can change n_neighbors to any desired integer. Here, we use 3.
    knn_model = train_knn_classifier(df, n_neighbors=3)
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
    dropout_predictions = predict_dropout(knn_model, new_students)
    
    # Display the predictions in a user-friendly format.
    for i, prediction in enumerate(dropout_predictions):
        status = "Dropout" if prediction == 1 else "No Dropout"
        print(f"Student {i+1}: Predicted status - {status}")

