# Syntecxhub Spam Detection CLI - Task 2

## Overview

This project was developed as part of the **SyntecxHub Machine Learning Internship (Week 2)**.

The application uses **Natural Language Processing (NLP)** and **Machine Learning** techniques to classify SMS messages as either **Spam** or **Ham (Legitimate)**. The system performs text preprocessing, feature extraction using TF-IDF, model training, evaluation, visualization, and real-time predictions through a Command Line Interface (CLI).

---

## Objectives

* Preprocess and clean text data
* Convert text into numerical features using TF-IDF
* Train and compare multiple machine learning models
* Evaluate model performance using classification metrics
* Visualize results using charts and word clouds
* Save trained models for future predictions
* Provide an interactive CLI for message classification

---

## Features

### Text Preprocessing

* Lowercase conversion
* URL removal
* Email removal
* Special character removal
* Stopword removal
* Lemmatization

### Feature Engineering

* TF-IDF Vectorization
* Unigram and Bigram Features

### Machine Learning Models

* Multinomial Naive Bayes
* Logistic Regression
* Linear Support Vector Machine (SVM)

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Classification Report
* Confusion Matrix

### Visualizations

* Confusion Matrix Heatmap
* Model Comparison Graph
* Spam Word Cloud
* Ham Word Cloud

### Additional Features

* Best Model Selection
* Pipeline Serialization using Joblib
* Prediction History Logging
* Interactive CLI Interface

---

## Project Structure

```text
Syntecxhub_Spam-detection-cli_task2/
│
├── config/
│   └── config.py
│
├── data/
│   ├── raw/
│   │   └── spam.csv
│   └── processed/
│
├── models/
│
├── outputs/
│   ├── logs/
│   ├── plots/
│   └── reports/
│
├── src/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── model_selection.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── save_pipeline.py
│   ├── train.py
│   └── visualize.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

**SMS Spam Collection Dataset**

Dataset Statistics:

* Total Messages: 5,572
* Ham Messages: 4,825
* Spam Messages: 747

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Syntecxhub_Spam-detection-cli_task2.git
```

Move into the project directory:

```bash
cd Syntecxhub_Spam-detection-cli_task2
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
python main.py
```

### CLI Menu

```text
========================================
ADVANCED SPAM DETECTION SYSTEM
========================================

1. Train Models
2. Evaluate Model
3. Generate Visualizations
4. Save Pipeline
5. Predict Message
6. Exit
```

---

## Sample Prediction

### Input

```text
Congratulations!
You have won ₹50,000.
Claim your reward now.
```

### Output

```text
Prediction : SPAM
```

### Input

```text
Hi Ganesh, can we meet tomorrow to discuss the project?
```

### Output

```text
Prediction : HAM
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* NLTK
* Matplotlib
* Seaborn
* WordCloud
* Joblib

---

## Learning Outcomes

Through this project, I gained practical experience in:

* Natural Language Processing (NLP)
* Text Feature Extraction
* Machine Learning Model Training
* Model Evaluation and Comparison
* Data Visualization
* Python Project Structuring
* CLI Application Development

---

## Future Enhancements

* Email Spam Detection
* Web-Based Interface using Flask
* Streamlit Dashboard
* Deep Learning-Based Classification
* REST API Deployment

---

## Author

**Ganesh Palav**

B.Tech Computer Science and Engineering


