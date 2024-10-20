# Flower Classification Using Logistic Regression

This project demonstrates how to use a Logistic Regression model to classify different species of flowers from the Iris dataset. The model predicts flower species based on four key features: sepal length, sepal width, petal length, and petal width. This notebook provides a step-by-step guide to implementing the model and visualizing the data.

## Dataset

The dataset used is the Iris dataset, which contains 150 samples of flowers belonging to three different species:
- **Iris-setosa**
- **Iris-versicolor**
- **Iris-virginica**

Each flower sample includes the following features:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

## Project Workflow

The notebook `Flower_Data_Set.ipynb` includes the following steps:

1. **Dataset Loading**: The Iris dataset is loaded using Pandas.
2. **Data Visualization**: Visualizations are created using Seaborn to explore the relationships between flower features.
3. **Label Mapping**: The species names are mapped to numerical values for classification.
4. **Data Splitting**: The dataset is split into training and test sets using `train_test_split`.
5. **Model Training**: A Logistic Regression model is trained on the training data.
6. **Model Evaluation**: The model is evaluated on the test set, and the accuracy is displayed.

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
print(f"Model Accuracy: {model.score(x_test, y_test):.2f}")
```

### Visualizations

One key visualization in this project is a scatter plot of petal length vs sepal width for different species:

```python
sns.FacetGrid(df, hue="Species", height=6).map(plt.scatter, "PetalLengthCm", "SepalWidthCm").add_legend()
```

This plot helps us understand the distribution of features and how they vary between different flower species.

## Requirements

To run this project, need following Python libraries:

- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Conclusion

This project provides a simple yet effective approach to classifying flowers using a Logistic Regression model. It emphasizes the importance of data visualization and model evaluation.

## Acknowledgments

- **Iris Dataset**: The dataset was sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris).
