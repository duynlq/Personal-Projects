import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer

from sklearn.metrics import confusion_matrix


def do_my_study():

    df = pd.read_csv('log2.csv')

    df = rename_cols(df)

    df = fix_response(df)

    df, categorical_vars = fix_cat_vars(df)

    X_train, X_test, y_train, y_test = model_prep(df)

    grid_search1 = do_SVM(categorical_vars, X_train, y_train)
    print("Best parameters: ", grid_search1.best_params_)
    print("Best score: ", grid_search1.best_score_)
    display_class_report(grid_search1, X_test, y_test)

    grid_search2 = do_SGD(categorical_vars, X_train, y_train)
    print("Best parameters: ", grid_search2.best_params_)
    print("Best score: ", grid_search2.best_score_)
    display_class_report(grid_search2, X_test, y_test)


def rename_cols(df):
    # Fix variable names
    df = df.rename(columns={"Source Port": "source_port",
                            "Destination Port": "destination_port",
                            "NAT Source Port": "nat_source_port",
                            "NAT Destination Port": "nat_destination_port",
                            "Action": "action",
                            "Bytes": "bytes",
                            "Bytes Sent": "bytes_sent",
                            "Bytes Received": "bytes_received",
                            "Packets": "packets",
                            "Elapsed Time (sec)": "elapsed_time_sec",
                            "pkts_sent": "packets_sent",
                            "pkts_received": "packets_received"})

    return df


def fix_response(df):

    # Combine "drop" and "reset" = "deny"
    df['action'] = df['action'].map({
        'allow': 'allow',
        'deny': 'deny',
        'reset-both': 'deny',
        'drop': 'deny'
        })
    # Convert to numeric and categorical
    df['action'] = df['action'].replace(
        ['allow', 'deny'], [1, 0]).astype('category')

    return df


def fix_cat_vars(df):

    categorical_vars = ['source_port', 'destination_port',
                        'nat_source_port', 'nat_destination_port']

    for var in categorical_vars:
        # Set frequencies of categories of a given categorical variable
        var_freqs = df[var].value_counts()

        # Replace categories with less than X frequency with 0
        vars_less_than_4 = var_freqs[var_freqs < 4].index
        df[var] = df[var].replace(vars_less_than_4, 0)

        # Convert to factors
        df[var] = df[var].astype("category")

    return df, categorical_vars


def model_prep(df):
    X = df.loc[:, df.columns != 'action']
    y = df['action']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def do_SVM(categorical_vars, X_train, y_train):

    numerical_vars = ['bytes', 'bytes_sent',
                      'bytes_received', 'packets',
                      'elapsed_time_sec', 'packets_sent',
                      'packets_received']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_vars),
            ('cat', categorical_transformer, categorical_vars)
        ])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', SVC())])

    param_grid = {
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': ['scale', 'auto']
    }

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=3)

    grid_search.fit(X_train, y_train)

    return grid_search


def do_SGD(categorical_vars, X_train, y_train):

    numerical_vars = ['bytes', 'bytes_sent',
                      'bytes_received', 'packets',
                      'elapsed_time_sec', 'packets_sent',
                      'packets_received']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_vars),
            ('cat', categorical_transformer, categorical_vars)
        ])

    pipeline = Pipeline(
        steps=[('preprocessor', preprocessor),
               ('classifier', SGDClassifier(early_stopping=True,
                                            loss='squared_hinge',
                                            n_jobs=-1,
                                            random_state=42
                                            ))])

    param_grid = {'classifier__penalty': ['l2', 'l1', 'elasticnet'],
                  'classifier__l1_ratio': [0.0, 0.1, 0.2, 0.3, 0.4,
                                           0.5, 0.6, 0.7, 0.8, 0.9,
                                           1.0]}

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=3)

    grid_search.fit(X_train, y_train)

    return grid_search


def display_class_report(model, X_test, y_test):
    y_preds = model.predict(X_test)
    # y_preds = (predicted_proba[:, 1] >= threshold).astype('int')

    print(  # "\nClassification Report \nThreshold = ", threshold,
          "\nAlgorithm = ", model, "):")
    tn, fp, fn, tp = confusion_matrix(y_test, y_preds).ravel()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    specificity = tn / (tn+fp)
    G_mean = np.sqrt(recall*specificity)
    AUC = (recall+specificity)/2
    print("Precision: \t", precision)
    print("Recall: \t", recall)
    print("Specificity: \t", specificity)
    print("G-mean: \t", G_mean)
    print("AUC: \t\t", AUC)


if __name__ == "__main__":

    do_my_study()
