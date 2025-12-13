import pandas as pd
import numpy as np

from helpers import (
    load_data,
    clean_data,
    split_data,
    model_evaluation,
)

from multiclass_classification import (
    direct_multiclass_train,
    get_binary_dataset,
    data_resampling,
    get_malicious_dataset_with_merged_webattacks,
    get_webattack_dataset,
    AI_driven_network_traffic_analysis,
)

LABEL_COL = "Label"


def load_and_prepare_full_dataset():
    csv_files = [
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
    ]

    dfs = []

    for fname in csv_files:
        df_raw = load_data(fname)

        # normalize column names (fixes 'Label ' etc.)
        df_raw.columns = df_raw.columns.astype(str).str.strip()

        # find label column case-insensitively
        col_map = {c.lower(): c for c in df_raw.columns}
        if "label" not in col_map:
            raise ValueError(
                f"No label column found in {fname}. "
                f"First columns: {list(df_raw.columns)[:20]}"
            )

        real_label_col = col_map["label"]

        labels = df_raw[real_label_col]

        # make labels safe: strip spaces, force string, drop missing
        labels = labels.astype(str).str.strip()
        labels = labels.replace({"nan": np.nan, "None": np.nan, "": np.nan})

        features = df_raw.drop(columns=[real_label_col])

        # keep only rows with a real label
        mask = labels.notna()
        labels = labels[mask]
        features = features.loc[mask]

        features_clean = clean_data(features)

        df_clean = features_clean.copy()
        df_clean[LABEL_COL] = labels.values

        dfs.append(df_clean)

    df_full = pd.concat(dfs, ignore_index=True)
    return df_full


def main():
    print("Loading and preparing dataset")
    df_full = load_and_prepare_full_dataset()

    print("Splitting into train and test")
    df_train, df_test = split_data(df_full, ratio=0.8)

    X_train_full = df_train.drop(columns=[LABEL_COL])
    y_train_full = df_train[LABEL_COL]
    X_test_full = df_test.drop(columns=[LABEL_COL])
    y_test_full = df_test[LABEL_COL]

    print("\nDirect multi class classification on all classes")
    # fast and reliable on large data
    model_direct = direct_multiclass_train("dt", X_train_full, y_train_full)
    model_evaluation(model_direct, X_test_full, y_test_full)

    print("\nFirst layer binary classification benign vs malicious")
    df_binary_train = get_binary_dataset(df_train)
    df_binary_test = get_binary_dataset(df_test)

    df_binary_train_balanced = data_resampling(df_binary_train)

    X_train_bin = df_binary_train_balanced.drop(columns=[LABEL_COL])
    y_train_bin = df_binary_train_balanced[LABEL_COL]
    X_test_bin = df_binary_test.drop(columns=[LABEL_COL])
    y_test_bin = df_binary_test[LABEL_COL]

    model_1st = direct_multiclass_train("dt", X_train_bin, y_train_bin)
    model_evaluation(model_1st, X_test_bin, y_test_bin)

    print("\nSecond layer multi class classification on malicious samples")
    df_mal_train = get_malicious_dataset_with_merged_webattacks(df_train)
    df_mal_test = get_malicious_dataset_with_merged_webattacks(df_test)

    X_train_mal = df_mal_train.drop(columns=[LABEL_COL])
    y_train_mal = df_mal_train[LABEL_COL]
    X_test_mal = df_mal_test.drop(columns=[LABEL_COL])
    y_test_mal = df_mal_test[LABEL_COL]

    model_2nd = direct_multiclass_train("dt", X_train_mal, y_train_mal)
    model_evaluation(model_2nd, X_test_mal, y_test_mal)

    print("\nThird layer web attack subtype classification")
    df_web_train = get_webattack_dataset(df_train)
    df_web_test = get_webattack_dataset(df_test)

    X_train_web = df_web_train.drop(columns=[LABEL_COL])
    y_train_web = df_web_train[LABEL_COL]
    X_test_web = df_web_test.drop(columns=[LABEL_COL])
    y_test_web = df_web_test[LABEL_COL]

    model_3rd = direct_multiclass_train("dt", X_train_web, y_train_web)
    model_evaluation(model_3rd, X_test_web, y_test_web)

    print("\nFinal hierarchical system evaluation")
    AI_driven_network_traffic_analysis(
        model_1st,
        model_2nd,
        model_3rd,
        X_test_full,
        y_test_full,
    )


if __name__ == "__main__":
    main()
