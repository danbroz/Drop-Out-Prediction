import pandas as pd
import numpy as np
from cassandra.cluster import Cluster

# ---------------------------------------------------------------------------
# ASSUMPTION: The following trained models are already available in the scope.
# They might have been trained in previous sessions. For example:
#
# knn_model = ...                   # K-Nearest Neighbors classifier
# svm_model = ...                   # Support Vector Machine classifier
# decision_tree_model = ...         # Decision Tree classifier
# logistic_model = ...              # Logistic Regression classifier
# naive_bayes_model = ...           # Naive Bayes classifier
# random_forest_model = ...         # Random Forest classifier
# nn_model = ...                    # Neural Network (MLP) binary classifier
# boosted_trees_model = ...         # Gradient Boosting classifier (boosted trees)
# sgd_model = ...                   # SGD classifier with oversampling
# regression_model = ...            # Gradient Boosting Regressor (for continuous dropout risk)
#
# For this example, we assume these models are already imported and trained.
# ---------------------------------------------------------------------------

def load_new_student_data_from_cassandra():
    """
    Connects to the Cassandra database and loads ~70 rows of new student data
    from the 'newStudents' table.
    
    Expected columns in the 'newStudents' table:
        - attendance: Numeric value indicating attendance rate or days attended.
        - prev_gpa: The student's previous year's GPA.
    
    Returns:
        new_student_df (pd.DataFrame): DataFrame with columns 'attendance' and 'prev_gpa'.
    """
    # Connect to the Cassandra cluster (adjust IP if necessary)
    cluster = Cluster(['127.0.0.1'])
    # Connect to the keyspace (assumed to be 'school_data')
    session = cluster.connect('school_data')
    
    # Define a query to fetch the new student data (70 rows)
    query = "SELECT attendance, prev_gpa FROM newStudents LIMIT 70"
    rows = session.execute(query)
    
    # Convert the rows to a list of dictionaries
    data = []
    for row in rows:
        data.append({
            'attendance': row.attendance,
            'prev_gpa': row.prev_gpa
        })
    
    # Shutdown the Cassandra connection
    session.shutdown()
    cluster.shutdown()
    
    # Return the data as a DataFrame
    return pd.DataFrame(data)

def aggregate_new_student_features(new_student_df):
    """
    Given a DataFrame with ~70 rows of data for a new student,
    aggregate the features to form a single feature vector.
    
    For simplicity, this function computes the mean of each feature.
    
    Parameters:
        new_student_df (pd.DataFrame): DataFrame with columns 'attendance' and 'prev_gpa'.
    
    Returns:
        feature_vector (list): A nested list [[avg_attendance, avg_prev_gpa]] suitable for model input.
    """
    avg_attendance = new_student_df['attendance'].mean()
    avg_prev_gpa = new_student_df['prev_gpa'].mean()
    # Return as a 2D array (one sample with 2 features)
    return [[avg_attendance, avg_prev_gpa]]

def get_model_predictions(feature_vector, models):
    """
    Call each model to predict the dropout risk for the new student.
    
    For classification models, we assume that a prediction of 1 indicates "at risk".
    For the regression model, we convert its continuous prediction to binary risk
    using a threshold (e.g., > 0.5 means at risk).
    
    Parameters:
        feature_vector (list): The aggregated features as a 2D list.
        models (dict): Dictionary mapping model names to trained model objects.
        
    Returns:
        predictions (dict): Dictionary mapping model names to a binary prediction (1: at risk, 0: not at risk).
    """
    predictions = {}
    for name, model in models.items():
        pred = model.predict(feature_vector)
        # If the model is the regression model, convert the continuous prediction to binary.
        if name == "Gradient Boosting Regressor":
            # For example, if predicted value > 0.5, then consider the student at risk.
            risk = 1 if pred[0] > 0.5 else 0
            predictions[name] = risk
        else:
            # For classifiers, assume the prediction is already binary (or can be converted to int)
            predictions[name] = int(pred[0])
    return predictions

def majority_vote(predictions):
    """
    Determine the final decision by majority vote.
    
    If more than half of the models predict "at risk" (i.e., 1), then final decision is "at risk".
    
    Parameters:
        predictions (dict): Dictionary of binary predictions.
        
    Returns:
        final_decision (int): 1 if at risk, 0 if not at risk.
    """
    votes = sum(predictions.values())
    total = len(predictions)
    return 1 if votes > total / 2 else 0

def main():
    # Step 1: Load the new student data from the Cassandra table 'newStudents'
    new_student_df = load_new_student_data_from_cassandra()
    
    # Aggregate the ~70 rows to form a single feature vector.
    feature_vector = aggregate_new_student_features(new_student_df)
    
    # Step 2: Create a dictionary of all trained models.
    # (Assuming these models are already defined in the environment.)
    trained_models = {
        "KNN": knn_model,
        "SVM": svm_model,
        "Decision Tree": decision_tree_model,
        "Logistic Regression": logistic_model,
        "Naive Bayes": naive_bayes_model,
        "Random Forest": random_forest_model,
        "Neural Network (Binary)": nn_model,
        "Boosted Trees": boosted_trees_model,
        "SGD Classifier": sgd_model,
        "Gradient Boosting Regressor": regression_model
    }
    
    # Step 3: Get predictions from each model for the new student.
    predictions = get_model_predictions(feature_vector, trained_models)
    
    # Display individual model predictions.
    print("Individual Model Predictions:")
    for name, pred in predictions.items():
        status = "At Risk" if pred == 1 else "Not At Risk"
        print(f"{name}: {status}")
    
    # Step 4: Determine final decision by majority vote.
    final_decision = majority_vote(predictions)
    print("\nFinal Decision based on majority vote:")
    if final_decision == 1:
        print("The student is predicted to be AT RISK of dropping out.")
    else:
        print("The student is predicted to be NOT at risk of dropping out.")

# Run the main function when this script is executed.
if __name__ == "__main__":
    main()

