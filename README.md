# CS-345-Final-Project
CS 345 Final Project
AI Driven Network Traffic Analysis

Overview
This project implements direct multiclass classification and a hierarchical
AI driven network traffic analysis system using the CIC IDS 2017 dataset.
The goal is to compare a flat multiclass model with a three layer hierarchical
approach for network intrusion detection.

Files
helpers.py
Data loading cleaning splitting and model evaluation functions

multiclass_classification.py
Implements direct multiclass classification and the hierarchical
three layer classification system

main.py
Runs all experiments and prints evaluation results

Dataset
CIC IDS 2017 CSV files provided with the assignment.
The label column is normalized automatically.

How to Run
Install dependencies
python3 -m pip install pandas numpy scikit-learn

Run
python3 main.py

Outputs
The program prints
Accuracy
Confusion matrices
Classification reports
for
Direct multiclass classification
Binary benign vs malicious classification
Malicious multiclass classification
Web attack subtype classification
Final hierarchical system

Results
The hierarchical system achieves approximately 99.8 percent overall accuracy.
Lower accuracy is observed for web attack subtypes due to severe class imbalance.

Notes
Results may vary slightly due to random sampling.

