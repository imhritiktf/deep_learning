# Iris Species Classification using Perceptron and Keras ANN

This notebook demonstrates the process of classifying Iris species using two different machine learning models: a simple Perceptron and a more complex Keras Artificial Neural Network (ANN). The steps cover data loading, preprocessing, model training, and evaluation.

## Table of Contents
1.  [Environment Setup](#environment-setup)
2.  [Data Preparation](#data-preparation)
3.  [Perceptron Model Implementation](#perceptron-model-implementation)
4.  [Keras Sequential Model (ANN) Implementation](#keras-sequential-model-ann-implementation)

---

## 1. Environment Setup

The notebook begins by importing necessary libraries for data manipulation, visualization, and machine learning:

*   **Data Handling:** `pandas`, `numpy`
*   **Visualization:** `matplotlib.pyplot`, `seaborn`
*   **Scikit-learn:** `train_test_split`, `LabelEncoder`, `StandardScaler`, `Perceptron`, `accuracy_score`, `classification_report`, `confusion_matrix`
*   **TensorFlow/Keras (Deep Learning):** `tensorflow`, `Sequential`, `Dense`, `Dropout`, `to_categorical`

---

## 2. Data Preparation

### Data Loading and Inspection

The Iris dataset, a classic dataset for classification, is loaded using `seaborn.load_dataset('iris')`.

Key inspection steps include:
*   `df.head()`: Displaying the first few rows of the dataset.
*   `df['species'].value_counts()`: Checking the distribution of target classes.
*   `df.info()`: Summarizing the DataFrame, including data types and non-null counts.

### Feature and Target Separation
Features (`X`) are separated from the target variable (`y`, the 'species' column).

### Label Encoding
The categorical `species` column (`y`) is converted into numerical integers (0, 1, 2) using `LabelEncoder`.

### Train-Test Split
The dataset is split into training and testing sets with an 80/20 ratio, ensuring stratified sampling to maintain class proportions, and a fixed `random_state` for reproducibility.

### Feature Scaling
`StandardScaler` is used to standardize the numerical features (`X_train` and `X_test`). This is crucial for models like Perceptron and ANNs, as it helps in faster convergence and better performance. `fit_transform` is applied only to `X_train` to prevent data leakage, and `transform` is applied to `X_test`.

---

## 3. Perceptron Model Implementation

### Model Training
A `Perceptron` classifier is initialized with `max_iter=100` and `random_state=42`, then trained on the scaled training data (`X_train_scaled`, `y_train`).

### Prediction and Evaluation
Predictions are made on the scaled test data (`X_test_scaled`), and the model's performance is evaluated using:
*   `accuracy_score`: Achieved **0.8667** (approximately 86.67%).
*   `classification_report`: Provides precision, recall, and F1-score for each class.
    *   Class 0 (setosa) showed perfect precision, recall, and f1-score (1.00).
    *   Classes 1 and 2 (versicolor and virginica) showed good but not perfect scores, indicating some misclassifications.
*   `confusion_matrix`: 
    ```
    [[10,  0,  0],
     [ 0,  7,  3],
     [ 0,  1,  9]]
    ```
    This matrix confirms that setosa was perfectly classified. There were 3 misclassifications for versicolor (predicted as virginica) and 1 misclassification for virginica (predicted as versicolor).

---

## 4. Keras Sequential Model (ANN) Implementation

### One-Hot Encoding
Before training the Keras model, the integer-encoded target labels (`y_train`, `y_test`) are converted to one-hot encoded categorical format using `to_categorical`.

### Model Architecture
A `Sequential` Keras model is defined with the following layers:
*   **Input Layer:** `Dense(16, activation='relu', input_shape=(4,))` - A dense layer with 16 neurons, ReLU activation, expecting 4 input features.
*   **Hidden Layer:** `Dense(8, activation='relu')` - Another dense layer with 8 neurons and ReLU activation.
*   **Output Layer:** `Dense(3, activation='softmax')` - A dense layer with 3 neurons (for 3 species classes) and softmax activation for multi-class probability distribution.

### Model Compilation
The model is compiled using:
*   **Optimizer:** `adam`
*   **Loss Function:** `categorical_crossentropy` (suitable for multi-class classification with one-hot encoded labels)
*   **Metrics:** `accuracy`

### Model Training
The model is trained for 100 epochs with a `batch_size` of 8 and a `validation_split` of 0.2. The `verbose=1` parameter provides a progress bar during training.

Initial training showed good convergence with no significant overfitting, with training and validation accuracy steadily increasing and loss decreasing over epochs.

### Visualization of Training History
Plots are generated to visualize:
*   **Model Accuracy Over Epochs:** Shows both training and validation accuracy trends.
*   **Model Loss Over Epochs:** Shows both training and validation loss trends.

These plots help in understanding the model's learning process and detecting potential overfitting or underfitting.

### Prediction and Evaluation

*   **Sample Predictions:** The model predicts the species for the first 5 test samples. The raw predictions (probabilities) are converted back to class labels and then to original species names using the `LabelEncoder`.

*   **Overall Test Set Evaluation:** The model's performance is evaluated on the full test set:
    *   `model.evaluate()`: Reports a test accuracy of approximately **0.9333** (93.33%) and a loss of 0.1102.
    *   `classification_report`:
        ```
        Keras Model Classification Report:
                      precision    recall  f1-score   support

                   0       1.00      1.00      1.00        10
                   1       0.90      0.90      0.90        10
                   2       0.90      0.90      0.90        10

                accuracy                           0.93        30
               macro avg       0.93      0.93      0.93        30
            weighted avg       0.93      0.93      0.93        30
        ```
        The Keras model shows improved performance compared to the Perceptron, with higher precision, recall, and F1-scores across all classes, particularly for 'versicolor' and 'virginica'.

    *   `confusion_matrix`:
        ```
        Keras Model Confusion Matrix:
        [[10  0  0]
         [ 0  9  1]
         [ 0  1  9]]
        ```
        This confusion matrix shows that class 0 (setosa) is still perfectly classified. For class 1 (versicolor), there was 1 misclassification (predicted as virginica), and for class 2 (virginica), there was 1 misclassification (predicted as versicolor). This indicates a slight improvement in reducing misclassifications between versicolor and virginica compared to the Perceptron.
