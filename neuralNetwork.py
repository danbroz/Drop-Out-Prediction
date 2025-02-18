# Import necessary libraries
import numpy as np                  # For numerical operations
import pandas as pd                 # For data manipulation
from cassandra.cluster import Cluster  # To connect and query the Cassandra database

# Import scikit-learn modules for our neural network model and data scaling
from sklearn.neural_network import MLPClassifier  # For the Neural Network classifier
from sklearn.preprocessing import StandardScaler    # For scaling input features
from sklearn.pipeline import make_pipeline          # To create a pipeline that chains scaling and modeling

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
    
    # Define the CQL query to select necessary columns from the 'students' table.
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

def train_neural_network_classifier(df):
    """
    Train a Neural Network classifier (MLPClassifier) using student data.
    
    The classifier uses the following features:
        - attendance
        - prev_gpa
    to predict the dropout status (0 or 1). A pipeline is used to scale the data before training.
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing the columns 'attendance', 'prev_gpa', and 'dropout'.
    
    Returns:
        model (Pipeline): The trained pipeline containing the StandardScaler and MLPClassifier.
    """
    # Extract features (attendance and previous GPA) as a NumPy array.
    X = df[['attendance', 'prev_gpa']].values
    
    # Extract the target dropout labels.
    y = df['dropout'].values
    
    # Create a pipeline that first scales the data and then trains a Neural Network.
    # Here, we use a single hidden layer with 10 neurons; you can adjust hidden_layer_sizes as needed.
    model = make_pipeline(
        StandardScaler(), 
        MLPClassifier(hidden_layer_sizes=(10,), random_state=42, max_iter=1000)
    )
    
    # Train the pipeline (scaling is applied, then the MLPClassifier is fitted)
    model.fit(X, y)
    
    # OPTIONAL: Evaluate the model's performance on the training data.
    train_accuracy = model.score(X, y)
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    
    # Return the trained pipeline.
    return model

def predict_dropout(model, new_data):
    """
    Predict the dropout status for new student data using the trained Neural Network classifier.
    
    Parameters:
        model (Pipeline): The trained pipeline with scaling and the Neural Network classifier.
        new_data (list or array): New data points where each point is a list of two features: [attendance, prev_gpa].
    
    Returns:
        predictions (list): Predicted dropout statuses (0 or 1) for each new data point.
    """
    # Convert new_data into a NumPy array in case it isn't already.
    new_data = np.array(new_data)
    
    # The pipeline automatically scales new_data and then predicts using the Neural Network.
    predictions = model.predict(new_data)
    
    # Return the predictions.
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
    # Step 2: Train the Neural Network classifier using the fetched data.
    # -------------------------------------------------------------------------
    print("\nTraining Neural Network classifier...")
    nn_model = train_neural_network_classifier(df)
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
    dropout_predictions = predict_dropout(nn_model, new_students)
    
    # Display the predictions in a user-friendly format.
    for i, prediction in enumerate(dropout_predictions):
        status = "Dropout" if prediction == 1 else "No Dropout"
        print(f"Student {i+1}: Predicted status - {status}")

