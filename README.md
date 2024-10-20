# Flower Classification Using Linear Regression

This project implements a whitebox machine learning model using linear regression to classify flowers from the Iris dataset. The model predicts flower species based on features like sepal length, sepal width, petal length, and petal width. It offers an interpretable and transparent approach to classification.

## Dataset

The dataset used is the famous Iris dataset, consisting of 150 instances and 4 features. It includes the following species:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

## Project Details

The Jupyter notebook `Flower_Data_Set.ipynb` performs the following steps:
1. Loads the dataset using Pandas.
2. Visualizes the data distribution using Seaborn.
3. Maps the flower species to numerical labels for model training.
4. Splits the dataset into training and test sets.
5. Trains a Logistic Regression model on the dataset.
6. Evaluates the model's accuracy on the test data.

### Code Example

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset
path = "/content/drive/MyDrive/Flower Dataset/Iris.csv"
df = pd.read_csv(path)

# Visualize the data
sns.FacetGrid(df, hue="Species", height=6).map(plt.scatter, "PetalLengthCm", "SepalWidthCm").add_legend()

# Map species names to numerical labels
flower_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df['Species'] = df['Species'].map(flower_mapping)

# Split the data
x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = df['Species'].values
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Train and evaluate the model
model = LogisticRegression()
model.fit(x_train, y_train)
print(f"Model Accuracy: {model.score(x_test, y_test)}")
