import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


LABEL_COL = "Label"

# 2.1 Direct Multi Class Classification
def direct_multiclass_train(model_name, X_train, y_train):
    """
    model_name  "mlp" or "dt"
    X_train     features DataFrame or array
    y_train     labels Series or array

    Constructs the chosen model, fits it, and returns the trained model.
    """
    if model_name == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=300,
            random_state=42
        )
    elif model_name == "dt":
        model = DecisionTreeClassifier(
            random_state=42
        )
    else:
        raise ValueError("Unsupported model_name use 'mlp' or 'dt'")

    model.fit(X_train, y_train)
    return model


# 2.2.1 First layer binary classification
def get_binary_dataset(df):
    """
    Takes original multi class DataFrame and returns a new DataFrame
    where the label column has only two values
    BENIGN and MALICIOUS

    All rows whose label is exactly "BENIGN" remain BENIGN
    All other labels become MALICIOUS
    """
    df_bin = df.copy()
    df_bin[LABEL_COL] = df_bin[LABEL_COL].apply(
        lambda x: "BENIGN" if x == "BENIGN" else "MALICIOUS"
    )
    return df_bin


def data_resampling(df_binary_train):
    """
    Takes a binary training DataFrame with labels BENIGN and MALICIOUS
    Reduces class imbalance using random undersampling of the majority class

    Steps
    separate features and labels
    undersample the majority class
    return resampled DataFrame with the same columns as input
    """
    df = df_binary_train.copy()

    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found")

    # Split by class
    benign = df[df[LABEL_COL] == "BENIGN"]
    malicious = df[df[LABEL_COL] == "MALICIOUS"]

    if len(benign) == 0 or len(malicious) == 0:
        raise ValueError("Both BENIGN and MALICIOUS must be present in df_binary_train")

    n_minority = min(len(benign), len(malicious))

    benign_sampled = benign.sample(n=n_minority, random_state=42)
    malicious_sampled = malicious.sample(n=n_minority, random_state=42)

    df_resampled = pd.concat([benign_sampled, malicious_sampled], axis=0)
    df_resampled = df_resampled.sample(frac=1.0, random_state=42).reset_index(drop=True)

    return df_resampled


# 2.2.2 Second layer malicious with merged web attacks
def get_malicious_dataset_with_merged_webattacks(df):
    """
    Takes original DataFrame df and returns a new DataFrame that
    contains only malicious samples

    All benign samples are removed
    Among the malicious samples the three web attack types are merged
    into a single "Web Attack" class label
    All other attack types stay as they are
    """
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found")

    df_mal = df.copy()

    # Remove benign
    df_mal = df_mal[df_mal[LABEL_COL] != "BENIGN"]

    # Known web attack labels in CIC IDS style datasets
    web_labels = {
        "Web Attack Brute Force",
        "Web Attack XSS",
        "Web Attack SQL Injection",
        "Web Attack"        # in case it already appears merged
    }

    def merge_label(lbl):
        if lbl in web_labels:
            return "Web Attack"
        return lbl

    df_mal[LABEL_COL] = df_mal[LABEL_COL].apply(merge_label)

    return df_mal


# 2.2.3 Third layer web attack subtypes
def get_webattack_dataset(df):
    """
    Takes a DataFrame df and returns a new DataFrame that contains
    only web attack samples

    The label column keeps the original three subtype labels
    "Web Attack Brute Force"
    "Web Attack XSS"
    "Web Attack SQL Injection"
    """
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found")

    web_subtypes = [
        "Web Attack Brute Force",
        "Web Attack XSS",
        "Web Attack SQL Injection"
    ]

    df_web = df[df[LABEL_COL].isin(web_subtypes)].copy()
    return df_web


# 2.2.4 Final hierarchical evaluation
def AI_driven_network_traffic_analysis(model_1st, model_2nd, model_3rd,
                                       X_test, y_test):
    """
    Combine three trained models into a hierarchical system

    model_1st  binary BENIGN vs MALICIOUS model
    model_2nd  multi class model on malicious with merged "Web Attack"
    model_3rd  multi class model on web attack subtypes

    X_test     original test features
    y_test     original test labels (with full class set)

    Prints final accuracy confusion matrix and classification report
    Returns array of final hierarchical predictions
    """
    # First layer predictions
    y_pred_first = model_1st.predict(X_test)

    # Ensure we can index consistently
    if hasattr(X_test, "index"):
        idx_list = list(X_test.index)
    else:
        idx_list = list(range(len(X_test)))

    final_pred = []

    for i, idx in enumerate(idx_list):
        first_label = y_pred_first[i]

        if first_label == "BENIGN":
            final_pred.append("BENIGN")
        else:
            # Second layer on malicious
            if hasattr(X_test, "loc"):
                x_single = X_test.loc[[idx]]
            else:
                x_single = X_test[i:i + 1, :]

            second_label = model_2nd.predict(x_single)[0]

            if second_label == "Web Attack":
                # Third layer on web attack subtype
                third_label = model_3rd.predict(x_single)[0]
                final_pred.append(third_label)
            else:
                final_pred.append(second_label)

    final_pred = np.array(final_pred)

    # Evaluation against original labels
    y_true = np.array(y_test)

    acc = accuracy_score(y_true, final_pred)
    print("Final hierarchical accuracy", acc)

    cm = confusion_matrix(y_true, final_pred)
    print("Final hierarchical confusion matrix")
    print(cm)

    report = classification_report(y_true, final_pred)
    print("Final hierarchical classification report")
    print(report)

    return final_pred
