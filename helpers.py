import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_data(fname):
    """
    Takes the path to a CSV file and loads it into a pandas DataFrame.
    """
    df = pd.read_csv(fname)
    return df


def clean_data(df):
    """
    Removes non numerical columns and replaces NaN and infinite values with column medians.
    """
    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])

    # Replace positive and negative infinity with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Replace NaN with column medians
    df.fillna(df.median(), inplace=True)

    return df


def split_data(df, ratio):
    """
    Randomly splits the DataFrame into training and testing sets
    according to the given ratio.
    ratio is the fraction of data used for training.
    """
    if not (0.0 < ratio < 1.0):
        raise ValueError("ratio must be between 0 and 1")

    df_train, df_test = train_test_split(
        df,
        train_size=ratio,
        shuffle=True,
        random_state=42
    )
    return df_train, df_test


def model_evaluation(model, X_test, y_test):
    """
    Evaluates a trained classification model on the testing set.
    Prints test accuracy, confusion matrix, and classification report.
    Returns the classification report string.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", acc)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    return report
