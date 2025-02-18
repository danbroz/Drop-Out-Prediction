# Drop-Out-Prediction
Predicts if a student will drop out using 16 different machine learning algorithms

Install requirements:

```pip install requirements.txt```

Load the data from CSV format to a Cassandra database:

```python loadData.py```

Train each model:

```python <model name>.py```

Load a new student data in CSV format called ```new_student_data.csv``` into a Cassandra database:

```python loadNewStudentTable.py```

Predict if a student is at risk of dropping out via any of the 16 models:

```python main.py```
