# Machine Learning for Intrusion Detection in Network Traffic

## Overview

This repository contains the codebase for my thesis on evaluating machine learning (ML) models for network intrusion detection.  
The research focuses on comparing supervised and unsupervised models to detect anomalies and malicious traffic in network data.  

Models included are:
- Autoencoder (TensorFlow)
- Random Forest (Scikit-learn)
- Support Vector Machine (Nystroem kernel approximation + LinearSVC)
- HDBSCAN clustering

## Project Structure

├── data  
│   ├── CIC-IDS2018  
│   └── processed  
├── models  
│   ├── autoencoder  
│   ├── hdbscan  
│   ├── random_forest  
│   └── svm  
├── scripts  
│   ├── preprocessing  
│   ├── training  
│   ├── evaluation  
│   └── utilities  
└── visualizations

## Key Components

### Data Preprocessing

- **Data Cleaning & Scaling**: Using `StandardScaler` for feature normalization.
- **Splits Generation**: Producing training, testing, unsupervised training, and zero-day attack datasets.

### Model Training

- **Autoencoder**: Neural network with bottleneck architecture for anomaly detection.
- **HDBSCAN**: Density-based clustering to identify outliers.
- **Random Forest**: Classifier optimized via hyperparameter tuning.
- **SVM**: Approximate RBF kernel implemented with Nystroem kernel approximation and LinearSVC.

### Model Evaluation

- **Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC.
- **ROC Curves**: Visual evaluation of model performance.
- **Zero-day Attack Detection**: Evaluating models' effectiveness against previously unseen threats.

### Scripts & Utilities

- Timer scripts for tracking long-running operations.
- Validation scripts for ensuring dataset integrity.

## Installation & Usage

### Clone repository

git clone https://github.com/simonflenman/intrusion-detection-thesis.git  
cd intrusion-detection-thesis

### Install dependencies:

pip install -r requirements.txt

### Preprocess data:

python src/data_preprocessing.py --mode [train|test|unsup_train|unsup_test|zero_day]

### Confirm data:

python src/data_conformation.py --mode [train|test|unsup_train|unsup_test|zero_day]

### Train models:

python src/train_autoencoder.py  
python src/train_rf.py  
python src/train_svm.py  
python src/train_hdbscan.py  

### Evaluate models:

python src/eval_autoencoder.py  
python src/eval_rf.py  
python src/eval_svm.py  
python src/eval_hdbscan.py  

## Technologies & Libraries

- **Languages**: Python

- **ML Frameworks**: TensorFlow, Scikit-learn, HDBSCAN

- **Data Handling**: Pandas, NumPy

- **Visualization**: Matplotlib

## Thesis Document
The full thesis can be accessed here:
