# Import necessary libraries
import numpy as np                  # For numerical operations
import pandas as pd                 # For data manipulation
from cassandra.cluster import Cluster  # To connect and query the Cassandra database

# Import modules for imbalanced classification, scaling, and SGD classifier
from imblearn.pipeline import make_pipeline  # For creating a pipeline that supports oversampling
from imblearn.over_sampling import SMOTE       # For oversampling the minority class
from sklearn.linear_model import SGDClassifier  # For the Stochastic Gradient Descent classifier
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

def train_stochastic_gradient_descent_classifier(df):
    """
    Train a classifier using Stochastic Gradient Descent (SGD) for imbalanced binary classification.
    
    The classifier uses the following features:
        - attendance
        - prev_gpa
    to predict the dropout status (0 or 1). Since the classes are imbalanced, SMOTE is used
    to oversample the minority class.
    
    A pipeline is used to:
        1. Scale the input features using StandardScaler.
        2. Apply SMOTE to oversample the minority class.
        3. Train an SGDClassifier using stochastic gradient descent.
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing 'attendance', 'prev_gpa', and 'dropout'.
    
    Returns:
        model (Pipeline): The trained pipeline including scaling, SMOTE, and the SGD classifier.
    """
    # Extract features (attendance and previous GPA) as a NumPy array.
    X = df[['attendance', 'prev_gpa']].values
    
    # Extract the target dropout labels.
    y = df['dropout'].values
    
    # Create a pipeline that scales data, applies SMOTE, and then trains the SGD classifier.
    # Here, we use the 'log' loss for logistic regression with SGD.
    model = make_pipeline(
        StandardScaler(),
        SMOTE(random_state=42),
        SGDClassifier(loss='log', random_state=42, max_iter=1000, tol=1e-3)
    )
    
    # Train the pipeline. Note: SM

