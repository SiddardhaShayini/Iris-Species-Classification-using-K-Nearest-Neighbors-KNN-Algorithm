# Iris Species Classification using K-Nearest Neighbors (KNN) Algorithm

## Project Overview

This project demonstrates the implementation of the **K-Nearest Neighbors (KNN)** classification algorithm to classify different species of Iris flowers. Using the well-known Iris dataset, we aim to build a robust model that can accurately predict the species of an Iris flower based on its sepal and petal measurements. This repository includes code for data preprocessing, model training, hyperparameter tuning, evaluation, and visualization of the results, including decision boundaries.

## Dataset

The project utilizes the **Iris dataset**, a classic and widely used dataset in machine learning for classification tasks. It contains 150 samples of Iris flowers, with three species (Iris-setosa, Iris-versicolor, Iris-virginica) equally distributed (50 samples each).

## Features and Target Variable

  - **Features (X):**

      - `SepalLengthCm`: Sepal length in centimeters.
      - `SepalWidthCm`: Sepal width in centimeters.
      - `PetalLengthCm`: Petal length in centimeters.
      - `PetalWidthCm`: Petal width in centimeters.

  - **Target (y):**

      - `Species`: The species of the Iris flower (Iris-setosa, Iris-versicolor, Iris-virginica). This categorical variable is encoded into numerical labels for model training.

## Key Steps and Techniques

1.  **Data Loading and Initial Exploration:**

      * Loading the Iris dataset using Pandas.
      * Displaying basic information and descriptive statistics of the dataset.

2.  **Data Preprocessing:**

      * Dropping irrelevant columns (e.g., 'Id').
      * **Encoding categorical target variable** ('Species') into numerical labels.
      * **Feature Scaling:** Applying `StandardScaler` to normalize the numerical features, which is crucial for distance-based algorithms like KNN.

3.  **Data Splitting:**

      * Splitting the dataset into training and testing sets (75% training, 25% testing) using `train_test_split` with `random_state` for reproducibility and `stratify` to maintain class distribution.

4.  **KNN Model Implementation and K-tuning:**

      * Implementing the `KNeighborsClassifier` from scikit-learn.
      * **Experimenting with different values of K** (number of neighbors) from 1 to 20 to find the optimal K that yields the highest accuracy.

5.  **Model Evaluation:**

      * Evaluating the KNN model using:
          * **Accuracy Score:** The proportion of correctly classified instances.
          * **Confusion Matrix:** A table that describes the performance of a classification model.
      * Visualizing the confusion matrix for the optimal K.

6.  **Visualizations:**

      * **Accuracy vs. K Value Plot:** A line plot showing how the model's accuracy changes with different K values, aiding in the selection of the optimal K.
      * **Decision Boundaries:** Visualizing the decision regions created by the KNN classifier using two key features (Petal Length and Petal Width). This helps understand how the model separates different classes.
      * **Pairplot:** A grid of scatter plots showing relationships between all pairs of features, colored by species, to understand feature distributions and correlations.
      * **Box Plots:** Displaying the distribution of each feature across different Iris species to highlight inter-species variations.

## Installation

To run this project, you'll need Python and the following libraries. You can install them using pip:

```bash
pip install pandas scikit-learn matplotlib seaborn numpy
```

## Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SiddardhaShayini/Iris-Species-Classification-using-K-Nearest-Neighbors-KNN-Algorithm.git
    cd Iris-Species-Classification-using-K-Nearest-Neighbors-KNN-Algorithm
    ```
2.  **Run the Jupyter Notebook**
    Use Google Colab or a Jupyter Notebook, simply open the `.ipynb` file and run all cells.

The script will output:

  * Dataset information and head.
  * Results of K-tuning, showing accuracy for each K.
  * The optimal K value and its corresponding accuracy.
  * The confusion matrix for the optimal K.
  * Several plots, including accuracy vs. K, confusion matrix heatmap, decision boundaries, pairplot, and box plots.

## Results and Visualizations

The project's output includes several informative plots. 

-----

## ✍️ Author

- Siddardha Shayini

-----
