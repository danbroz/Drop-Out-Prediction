# Import necessary libraries
import numpy as np                  # For numerical operations
import pandas as pd                 # For data manipulation
from cassandra.cluster import Cluster  # To connect and query the Cassandra database

# Import scikit-learn modules for our neural network model, scaling, and multi-label classification
from sklearn.neural_network import MLPClassifier  # For the Neural Network classifier
from sklearn.preprocessing import StandardScaler    # For scaling input features
from sklearn.pipeline import make_pipeline          # To create a pipeline for scaling and modeling
from sklearn.multioutput import MultiOutputClassifier  # To enable multi-label classification

def fetch_data_from_cassandra():
    """
    Connect to the Cassandra database and fetch 70 rows of student data.
    
    Expected columns in the 'students' table:
        - attendance: Numeric value indicating attendance rate or days attended.
        - prev_gpa: The student's previous year's GPA.
        - dropout: Binary indicator (1 means the student dropped out, 0 means they did not).
        - at_risk: Binary indicator (1 means the student is at risk, 0 means they are not).
    
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
    query = "SELECT attendance, prev_gpa, dropout, at_risk FROM students LIMIT 70"
    rows = session.execute(query)
    
    # Convert the Cassandra rows into a list of dictionaries.
    data = []
    for row in rows:
        data.append({
            'attendance': row.attendance,
            'prev_gpa': row.prev_gpa,
            'dropout': row.dropout,
            'at_risk': row.at_risk
        })
    
    # Shutdown the session and cluster connection once the data is fetched.
    session.shutdown()
    cluster.shutdown()
    
    # Create a pandas DataFrame from the data list for easier manipulation.
    df = pd.DataFrame(data)
    return df

def train_neural_network_multi_label_classifier(df):
    """
    Train a multi-label Neural Network classifier using student data.
    
    The classifier uses the following features:
        - attendance
        - prev_gpa
    to predict two labels:
        - dropout
        - at_risk
    A pipeline is used to first scale the data and then apply a MultiOutputClassifier
    that wraps an MLPClassifier for multi-label classification.
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing the columns 'attendance', 'prev_gpa', 'dropout', and 'at_risk'.
    
    Returns:
        model (Pipeline): The trained pipeline containing the StandardScaler and MultiOutputClassifier.
    """
    # Extract features (attendance and previous GPA) as a NumPy array.
    X = df[['attendance', 'prev_gpa']].values
    
    # Extract the target multi-labels as a NumPy array.
    # Here, y will have two columns: one for 'dropout' and one for 'at_risk'
    y = df[['dropout', 'at_risk']].values
    
    # Create a pipeline that first scales the data and then trains a multi-label neural network.
    # MultiOutputClassifier wraps MLPClassifier to handle multi-label classification.
    model = make_pipeline(
        StandardScaler(),
        MultiOutputClassifier(
            MLPClassifier(hidden_layer_sizes=(10,), random_state=42, max_iter=1000)
        )
    )
    
    # Train the pipeline on the dataset.
    model.fit(X, y)
    
    # OPTIONAL: Evaluate the model's performance on the training data.
    # The score will be the average accuracy across the two labels.
    train_accuracy = model.score(X, y)
    print(f"Training accuracy (average across labels): {train_accuracy * 100:.2f}%")
    
    # Return the trained pipeline.
    return model

def predict_labels(model, new_data):
    """
    Predict the multi-label outputs for new student data using the trained neural network.
    
    Parameters:
        model (Pipeline): The trained pipeline with scaling and the multi-label neural network classifier.
        new_data (list or array): New data points where each point is a list of two features: [attendance, prev_gpa].
    
    Returns:
        predictions (ndarray): Predicted multi-label outputs for each new data point.
                               Each prediction is an array, e.g., [dropout, at_risk].
    """
    # Convert new_data into a NumPy array for consistency.
    new_data = np.array(new_data)
    
    # The pipeline automatically scales new_data and then predicts the multi-label outputs.
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
    # Step 2: Train the multi-label Neural Network classifier using the fetched data.
    # -------------------------------------------------------------------------
    print("\nTraining multi-label Neural Network classifier...")
    multi_label_nn_model = train_neural_network_multi_label_classifier(df)
    print("Training complete!")
    
    # -------------------------------------------------------------------------
    # Step 3: Use the trained model to predict labels for new students.
    # -------------------------------------------------------------------------
    # Example new student data:
    # Each sub-list represents [attendance, prev_gpa].
    new_students = [
        [90, 3.5],  # Student with high attendance and a strong GPA.
        [60, 2.0],  # Student with lower attendance and a lower GPA.
        [80, 3.0]   # Student with moderate attendance and GPA.
    ]
    
    print("\nPredicting multi-label outputs for new student data...")
    predictions = predict_labels(multi_label_nn_model, new_students)
    
    # Display the predictions in a user-friendly format.
    # Each prediction is an array where:
    #   prediction[0] is the predicted 'dropout' status,
    #   prediction[1] is the predicted 'at_risk' status.
    for i, pred in enumerate(predictions):
        dropout_status = "Dropout" if pred[0] == 1 else "No Dropout"
        at_risk_status = "At Risk" if pred[1] == 1 else "Not At Risk"
        print(f"Student {i+1}: Predicted -> {dropout_status}, {at_risk_status}")

