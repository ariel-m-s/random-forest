# Random Forest implementation in Python

## Overview

This repository contains a Python implementation of a Random Forest model. The Random Forest algorithm is an ensemble learning method that combines multiple decision trees to enhance predictive accuracy and control overfitting.

## Files and structure

The repository includes the following files:

1. `randomforest/__init__.py`

This file initializes the Random Forest model, imports necessary components, and defines the main `RandomForestModel` class. The class includes methods for model training, prediction, and asserting predictions on a dataset.

2. `randomforest/decisiontree.py`

This file contains the implementation of the Decision Tree, a fundamental component of the Random Forest model. The Decision Tree recursively partitions the dataset based on feature values, creating a tree structure that facilitates predictive modeling.

3. `randomforest/preprocess.py`

This file provides utility functions for loading datasets and creating random samples for training the Random Forest model. The `load_dfs` function loads a dataset from a CSV file, and `random_sample` generates a random subdataset for training.

4. `main.ipynb`

Jupyter Notebook demonstrating the usage of the Random Forest model. It serves as a visual guide and provides insights into the training process.

## Example usage

```python
from randomforest import RandomForestModel

model = RandomForestModel()
model.fit(train_data, target_column)
predictions = model.predict(test_data)
accuracy = model.assert_predictions(test_data)
print(f"Model Accuracy: {accuracy}")
```

## Model parameters

The `fit` method in the `RandomForestModel` class accepts several parameters allowing model customization. These include:

- `n_estimators`: Number of trees in the forest.
- `frac_shape`: Proportion of dimensions for the random sample used in each tree.
- `max_depth`: Maximum depth of each decision tree.
- `min_samples_split`: The minimum number of samples a non-leaf node can have to split.

Adjusting the parameters of the Random Forest model can have a significant impact on its performance and behavior. It's common practice to perform a hyperparameter search using techniques like grid search or random search to find the optimal combination of parameters that maximizes the model's accuracy on a validation set.

## Visualization

The `generate_treemap` method in the `DecisionTreeModel` class creates an interactive treemap visualization of the decision tree. This visualization can help understand the structure of individual decision trees within the Random Forest.
