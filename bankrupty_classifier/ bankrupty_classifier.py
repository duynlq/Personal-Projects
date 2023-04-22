import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
# from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


def do_my_study():

    files = [
        '1year.arff'  # ,
        # '2year.arff'  # ,
        # '3year.arff' # ,
        # '4year.arff',  # ,
        # '5year.arff'  # ,
    ]

    df = import_data(files)

    impute_strategy = "median"
    resample_method = SMOTE(random_state=42)

    X_train, X_test, y_train, y_test = preproc(df, impute_strategy)

    # SMOTEEN (SMOTE and Edited Nearest Neighbours)
    # SMOTETomek (SMOTE and Tomek links)
    # X_train, y_train = resample(resample_method, X_train, y_train)

    rf_model = do_RF(X_train, y_train)
    display_AUC(rf_model, X_train, y_train)

    # threshold = .21
    # display_class_report(threshold, rf_model, X_test, y_test)
    # # print(Counter(y_train))


def import_data(files):

    df = pd.DataFrame(arff.loadarff(files[0])[0])

    for f in files[1:]:
        data_temp = arff.loadarff(f)
        df_temp = pd.DataFrame(data_temp[0])
        df = df.merge(df_temp, how='outer')

    return df


def preproc(df, impute_strategy):
    # Fixing response variable
    df['class'] = df['class'].replace([b'0', b'1'], [0, 1])

    # Splitting data
    X = df.loc[:, df.columns != 'class'].values
    y = df['class'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Impute
    X_train = SimpleImputer(strategy=impute_strategy).fit_transform(X_train)
    X_test = SimpleImputer(strategy=impute_strategy).fit_transform(X_test)

    # Normalize the data
    X_train = RobustScaler().fit_transform(X_train)
    X_test = RobustScaler().fit_transform(X_test)

    return X_train, X_test, y_train, y_test


def resample(method, X_train, y_train):
    method_holder = method
    X_train, y_train = method_holder.fit_resample(X_train, y_train)

    return X_train, y_train


def do_RF(X_train, y_train):
    rf_model = RandomForestClassifier(
        # CONTROL
        n_estimators=12,
        bootstrap=True,
        criterion="gini",
        max_depth=None,
        max_features=10,
        min_samples_leaf=3,
        min_samples_split=6)

    # SMOTEENN
    rf_model = RandomForestClassifier(
        n_estimators=20, bootstrap=False, criterion="entropy", max_depth=None,
        max_features=10, min_samples_leaf=4, min_samples_split=10)

    # SMOTETomek
    rf_model = RandomForestClassifier(
        n_estimators=15, bootstrap=True, criterion="gini", max_depth=None,
        max_features=9, min_samples_leaf=1, min_samples_split=7)

    rf_model.fit(X_train, y_train)

    return rf_model


def display_AUC(model, X_train, y_train):
    AUC_holder = cross_val_score(
        model, X_train, y_train, scoring="roc_auc", cv=10)
    mean_score = AUC_holder.mean()
    std_score = AUC_holder.std()
    print("\nMean AUC: {0:.3f} \nStd AUC: {1:.3f}".format(
        mean_score, std_score))


def display_class_report(threshold, model, X_test, y_test):
    predicted_proba = model.predict_proba(X_test)
    y_preds = (predicted_proba[:, 1] >= threshold).astype('int')

    print("\nClassification Report \nThreshold = ", threshold,
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
