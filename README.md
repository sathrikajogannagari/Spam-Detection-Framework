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

The project uses `features.txt`, which contains six numerical input features and one target column named `class`.

The dataset contains 36 records with the following features:

- Replies
- Behaviour
- Following
- Followers
- Burstiness
- NegativeRatio

The class labels are represented as:

- `0` – Non-Spam
- `1` – Spam

The labels are taken directly from the supplied dataset. Their original assignment method has not yet been independently verified.

## Model Evaluation

The models are evaluated using the following performance metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

A comparison graph is generated to display the performance of the four classifiers.

**Note:** The dataset contains only 36 records. Therefore, the evaluation results are preliminary and may vary significantly with different train/test splits. They should not be considered evidence of general model performance.

## Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/sathrikajogannagari/Spam-Detection-Framework.git
```

### 2. Navigate to the Project Folder

```bash
cd Spam-Detection-Framework
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python spam_detection.py
```

### 5. Use the Application

1. Upload the `features.txt` dataset.
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
