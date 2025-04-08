
# Banknote Authentication Classifier

## Overview

This project focuses on predicting whether a banknote is genuine or forged using machine learning models. The dataset includes features extracted from images of banknotes, such as the variance, skewness, and the interaction between variance and skewness. Various machine learning models, including **Logistic Regression** and **Decision Tree Classifier**, are used to evaluate the performance of different feature sets for binary classification.

## Features

- **variance**: Wavelet image variance
- **skewness**: Skewness of the image wavelet distribution
- **var_skew_interaction**: Interaction term between variance and skewness

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Model Performance](#model-performance)
- [Challenges Faced](#challenges-faced)
- [Future Additions](#future-additions)
- [Reflection](#reflection)

## Installation

To run this project, you need the following dependencies:

- Python 3.7 or higher
- Jupyter Notebook (optional, for code execution)
- Required libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

You can install the required libraries by running:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset used in this project is a collection of features extracted from images of genuine and forged banknotes. Each row represents a single banknote, with various features describing its image properties.

### Columns:
- **variance**: Wavelet image variance
- **skewness**: Skewness of the image wavelet distribution
- **curtosis**: curtosis of the image 
- **entropy**: entropy of the image
- **var_skew_interaction**: Interaction term between variance and skewness
- **class**: Binary target variable (0 for genuine, 1 for forged)

## Usage

1. **Feature Selection and Preprocessing**:  
   The features are selected based on their relevance to detecting forged banknotes. Features such as variance and skewness are used, with transformations like interaction terms and non-linear transformations of skewness to capture complex patterns.

2. **Model Training**:  
   The models used include **Logistic Regression** and **Decision Tree**. The data is split into training and testing sets (80%/20% split). The models are trained on different feature sets, and the performance is evaluated using accuracy, precision, recall, F1 score, and confusion matrix.

3. **Evaluate Models**:  
   For each feature set, the models are evaluated using the function `evaluate_feature_logr` for Logistic Regression and `evaluate_feature_dectree` for Decision Trees.

4. **Plotting**:  
   Visualizations such as scatter plots of `variance` vs. `skewness` are included to understand the relationship between features and the target class.

## Evaluation

The performance of the models is evaluated using the following metrics:
- **Accuracy**: Percentage of correct predictions
- **Precision**: Proportion of true positives out of all predicted positives
- **Recall**: Proportion of true positives out of all actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Breakdown of true positives, false positives, true negatives, and false negatives

### Example Results:

For the **Logistic Regression** model:

- **Case 1: Variance**  
  - Accuracy: 86.18%
  - Precision: 83.87%
  - Recall: 85.25%
  - F1 Score: 84.55%

- **Case 2: Skewness**  
  - Accuracy: 66.91%
  - Precision: 65.66%
  - Recall: 53.28%
  - F1 Score: 58.82%

- **Combo (Variance + Skewness + Interaction)**  
  - Accuracy: 90.55%
  - Precision: 88.71%
  - Recall: 90.16%
  - F1 Score: 89.43%

For the **Decision Tree** model, similar results are provided with comparisons for each feature set.

## Model Performance

- Logistic Regression generally performs better in terms of **precision**, making it a better choice when minimizing false positives is the priority.
- Decision Tree performs better in **recall**, making it useful in applications where identifying true positives (forged banknotes) is more important.
- **Combo Case (Variance + Skewness + Interaction)** shows that both models achieve similar performance (around 90% accuracy), but Logistic Regression has a slightly higher precision, while Decision Tree has higher recall.

## Challenges Faced

- **Feature Engineering**: Choosing the right features was a challenge, especially with interaction terms like **var_skew_interaction**, which didn’t perform well on its own.
- **Class Imbalance**: While stratified sampling was used, class imbalance could still affect model performance. Exploring techniques like **SMOTE** could address this in future work.
- **Model Overfitting**: The Decision Tree showed signs of overfitting, particularly with fewer features. Pruning or using ensemble methods like **Random Forest** could help mitigate this.
- **Evaluation Metrics**: Using multiple evaluation metrics (accuracy, precision, recall, F1 score) was essential, as relying on a single metric could misrepresent model performance, especially with imbalanced data.

## Future Additions

1. **Hyperparameter Tuning**:  
   Tuning hyperparameters for both models using grid search or random search to improve performance.

2. **Ensemble Methods**:  
   Exploring **Random Forests** or **Gradient Boosting** to reduce overfitting and improve generalization.

3. **Advanced Feature Engineering**:  
   Investigating additional features or transformations (log, square, or interaction terms) to improve model accuracy.

4. **Address Class Imbalance**:  
   Experimenting with oversampling/undersampling techniques, such as **SMOTE**, to handle class imbalance more effectively.

5. **Deep Learning**:  
   Exploring **Neural Networks** or other deep learning models could further improve classification performance, especially with more data.

## Reflection

This project helped me learn the importance of **feature selection** and how different features affect the performance of machine learning models. The project also emphasized the importance of balancing **precision and recall** when choosing a model and evaluation metrics. Testing multiple models and understanding their strengths was essential to determine the best classifier for the task at hand.
