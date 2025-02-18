# Import necessary libraries
import numpy as np                  # For numerical operations
import pandas as pd                 # For data manipulation
from cassandra.cluster import Cluster  # To connect and query the Cassandra database

# Import modules for imbalanced classification and model training
from imblearn.pipeline import make_pipeline  # To create a pipeline that supports oversampling
from imblearn.over_sampling import SMOTE       # For oversampling the minority class
from sklearn.neural_network import MLPClassifier  # For the Neural Network classifier
from sklearn.preprocessing import StandardScaler    # For scaling input features

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
    # Replace '127.0.0.1' with your Cassandra node's IP address if needed.
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
    
    # Create a pandas DataFrame from the list for easier manipulation.
    df = pd.DataFrame(data)
    return df

def train_imbalanced_classifier(df):
    """
    Train a Neural Network classifier for imbalanced binary classification using student data.
    
    The classifier uses the following features:
        - attendance
        - prev_gpa
    to predict the dropout status (0 or 1). Since the classes are imbalanced, SMOTE is used to 
    oversample the minority class.
    
    A pipeline is used to:
        1. Scale the input features using StandardScaler.
        2. Apply SMOTE to oversample the minority class.
        3. Train an MLPClassifier (Neural Network).
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing 'attendance', 'prev_gpa', and 'dropout'.
    
    Returns:
        model (Pipeline): The trained pipeline including scaling, SMOTE, and the classifier.
    """
    # Extract features (attendance and previous GPA) as a NumPy array.
    X = df[['attendance', 'prev_gpa']].values
    
    # Extract the target dropout labels.
    y = df['dropout'].values
    
    # Create a pipeline that scales data, applies SMOTE, and then trains the Neural Network.
    # SMOTE helps balance the minority class by generating synthetic examples.
    model = make_pipeline(
        StandardScaler(),
        SMOTE(random_state=42),
        MLPClassifier(hidden_layer_sizes=(10,), random_state=42, max_iter=1000)
    )
    
    # Train the pipeline. Note: SMOTE is applied only to the training data.
    model.fit(X, y)
    
    # OPTIONAL: Evaluate the model's performance on the training data.
    train_accuracy = model.score(X, y)
    print(f"Training accuracy (with SMOTE oversampling): {train_accuracy * 100:.2f}%")
    
    # Return the trained pipeline.
    return model

def predict_dropout(model, new_data):
    """
    Predict the dropout status for new student data using the trained imbalanced classifier.
    
    Parameters:
        model (Pipeline): The trained pipeline with scaling, SMOTE (used only during training), and the Neural Network.
        new_data (list or array): New data points where each point is a list of two features: [attendance, prev_gpa].
    
    Returns:
        predictions (list): Predicted dropout statuses (0 or 1) for each new data point.
    """
    # Convert new_data into a NumPy array for consistency.
    new_data = np.array(new_data)
    
    # The pipeline automatically scales the new data and predicts using the Neural Network.
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
    # Step 2: Train the imbalanced classifier using the fetched data.
    # -------------------------------------------------------------------------
    print("\nTraining Neural Network classifier with SMOTE for imbalanced classification...")
    imbalanced_model = train_imbalanced_classifier(df)
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
    dropout_predictions = predict_dropout(imbalanced_model, new_students)
    
    # Display the predictions in a user-friendly format.
    for i, prediction in enumerate(dropout_predictions):
        status = "Dropout" if prediction == 1 else "No Dropout"
        print(f"Student {i+1}: Predicted status - {status}")

