# Import necessary libraries
import numpy as np          # For numerical operations
import pandas as pd         # For data manipulation
from cassandra.cluster import Cluster  # To connect and query the Cassandra database
from sklearn.cluster import KMeans     # To perform K-Means clustering

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

def train_kmeans_classifier(df):
    """
    Train a K-Means classifier using student data.
    The K-Means algorithm will group students into 2 clusters based on the features:
        - attendance
        - prev_gpa
    Since we already have dropout labels (0 or 1), we can determine which cluster corresponds
    to dropout by checking the majority label within each cluster.
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing the columns 'attendance', 'prev_gpa', and 'dropout'.
    
    Returns:
        kmeans (KMeans): The trained KMeans model.
        cluster_to_dropout (dict): A mapping from each K-Means cluster label (0 or 1) to the predicted dropout status (0 or 1).
    """
    # Extract features from the DataFrame (attendance and previous GPA).
    X = df[['attendance', 'prev_gpa']].values  # Convert to a NumPy array.
    
    # Extract the true dropout labels.
    y = df['dropout'].values
    
    # Create the K-Means model with 2 clusters (since dropout is a binary outcome).
    # Setting random_state ensures reproducibility.
    kmeans = KMeans(n_clusters=2, random_state=42)
    
    # Fit the model to the data and predict cluster labels for each student.
    cluster_labels = kmeans.fit_predict(X)
    
    # Create a dictionary to map each cluster to a dropout prediction.
    cluster_to_dropout = {}
    
    # For each cluster, determine the majority dropout label.
    for cluster in np.unique(cluster_labels):
        # Get the indices of data points that belong to the current cluster.
        indices = np.where(cluster_labels == cluster)
        
        # Calculate the mean of the dropout labels in this cluster.
        # Because dropout labels are 0 or 1, a mean >= 0.5 suggests that most students dropped out.
        mean_dropout = np.mean(y[indices])
        
        # Determine the predicted dropout label: 1 if mean >= 0.5, otherwise 0.
        predicted_label = 1 if mean_dropout >= 0.5 else 0
        
        # Store the mapping in the dictionary.
        cluster_to_dropout[cluster] = predicted_label
        
        # Print the mapping details for debugging purposes.
        print(f"Cluster {cluster}: Mean dropout = {mean_dropout:.2f}, Assigned label = {predicted_label}")
    
    # Return both the trained model and the cluster-to-dropout mapping.
    return kmeans, cluster_to_dropout

def predict_dropout(kmeans, cluster_to_dropout, new_data):
    """
    Predict the dropout status for new student data using the trained K-Means classifier.
    
    Parameters:
        kmeans (KMeans): The trained KMeans model.
        cluster_to_dropout (dict): Mapping from cluster label to dropout status.
        new_data (list or array): New data points where each point is a list of two features: [attendance, prev_gpa].
    
    Returns:
        predictions (list): Predicted dropout statuses (0 or 1) for each new data point.
    """
    # Convert new_data into a NumPy array in case it isn't one already.
    new_data = np.array(new_data)
    
    # Use the KMeans model to predict which cluster each new data point belongs to.
    predicted_clusters = kmeans.predict(new_data)
    
    # Map the cluster predictions to dropout predictions using our dictionary.
    predictions = [cluster_to_dropout[cluster] for cluster in predicted_clusters]
    
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
    # Step 2: Train the K-Means classifier using the fetched data.
    # -------------------------------------------------------------------------
    print("\nTraining KMeans classifier...")
    kmeans_model, cluster_mapping = train_kmeans_classifier(df)
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
    dropout_predictions = predict_dropout(kmeans_model, cluster_mapping, new_students)
    
    # Display the predictions in a user-friendly format.
    for i, prediction in enumerate(dropout_predictions):
        status = "Dropout" if prediction == 1 else "No Dropout"
        print(f"Student {i+1}: Predicted status - {status}")

