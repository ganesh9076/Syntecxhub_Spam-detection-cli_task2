# Advanced SMS Spam Detection System

## Overview

This project is an Advanced SMS Spam Detection System built using Natural Language Processing (NLP) and Machine Learning. The system classifies incoming SMS messages as either **Spam** or **Ham (Legitimate)** using TF-IDF feature extraction and multiple machine learning models.

The project includes data preprocessing, feature engineering, model comparison, evaluation metrics, visualization, model persistence, and an interactive Command Line Interface (CLI) for real-time predictions.

---

## Features

* Text preprocessing and cleaning
* Stopword removal and lemmatization
* TF-IDF vectorization with n-grams
* Multiple model training and comparison

  * Naive Bayes
  * Logistic Regression
  * Linear SVM
* Automatic best model selection
* Accuracy, Precision, Recall, and F1-Score evaluation
* Confusion Matrix visualization
* Model Comparison visualization
* Spam and Ham Word Clouds
* Saved Machine Learning Pipeline
* Interactive CLI-based prediction system
* Prediction history logging

---

## Project Structure

```text
Syntecxhub_Spam-detection-cli/
│
├── config/
│   └── config.py
│
├── data/
│   └── raw/
│       └── spam.csv
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
└── README.md
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

## Dataset

SMS Spam Collection Dataset

* Total Messages: 5,572
* Ham Messages: 4,825
* Spam Messages: 747

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/advanced-spam-detection-cli.git
```

Navigate to the project directory:

```bash
cd advanced-spam-detection-cli
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

```bash
python main.py
```

CLI Menu:

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

## Evaluation Metrics

The system evaluates the best-performing model using:

* Accuracy
* Precision
* Recall
* F1 Score
* Classification Report
* Confusion Matrix

---

## Sample Prediction

Input:

```text
Congratulations! You have won ₹50,000. Claim your reward now.
```

Output:

```text
Prediction : SPAM
```

Input:

```text
Hi Ganesh, can we meet tomorrow to discuss the project?
```

Output:

```text
Prediction : HAM
```

---

## Visualizations

The project generates:

* Confusion Matrix
* Model Comparison Chart
* Spam Word Cloud
* Ham Word Cloud

All generated visualizations are stored in the `outputs/plots` directory.

---

## Future Enhancements

* Email Spam Detection
* Flask Web Application
* Streamlit Dashboard
* Deep Learning Models
* Real-Time Message Classification API

---

## Author

Ganesh Palav

B.Tech Computer Science and Engineering

