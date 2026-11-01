# Spam Detection Framework Using Machine Learning

## Project Overview

The Spam Detection Framework is a Python-based machine learning application designed to classify Twitter-related data into spam and non-spam categories using numerical features and supervised learning algorithms.

The application provides a graphical user interface (GUI) built using Tkinter and compares the performance of multiple machine learning classifiers.

## Objectives

- Develop a machine learning framework for spam classification.
- Apply different classification algorithms.
- Evaluate and compare model performance.
- Display evaluation results through a graphical interface.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Tkinter

## Machine Learning Algorithms

The following algorithms are implemented:

1. Random Forest Classifier
2. Support Vector Machine (SVM)
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes

## Dataset

The project uses `features.txt`, which contains six numerical input features and a target column named `class`.

The dataset contains 36 records. The class labels are represented as:
- 0 – Non-Spam
- 1 – Spam

The original source and method used to assign these labels should be verified before interpreting the results as validated spam classifications.

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

A comparison graph is generated to display classifier performance.

**Note:** The dataset is small (36 records), so evaluation results are preliminary and should not be considered evidence of general model performance.

## Installation and Execution

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd Spam-Detection-Framework
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python spam_detection.py
```

### 5. Use the application

1. Upload `features.txt`.
2. Prepare the training and testing data.
3. Train and evaluate all four classifiers.
4. View the model comparison graph.

## Project Structure

```text
Spam-Detection-Framework/
│
├── spam_detection.py
├── SpamDetection_original.py
├── features.txt
├── requirements.txt
└── README.md
```

## Future Enhancements

- Increase the size and diversity of the labelled dataset.
- Verify and improve ground-truth labels.
- Add text-based feature extraction.
- Perform cross-validation and more robust model evaluation.
- Improve the graphical user interface.

## Author

Sathrika Jogannagari