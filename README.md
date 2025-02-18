# Drop-Out-Prediction

Install requirements:

```pip install requirements.txt```

Load the data from CSV format to a Cassandra database:

```python loadData.py```

Train each model:

```python <model name>.py```

Load a new student data in CSV format called ```new_student_data.csv``` into a Cassandra database:

```python loadNewStudentTable.py```

Predict if a student is at risk of dropping out via any of the 12 models:

```python main.py```
