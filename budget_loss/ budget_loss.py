import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import InputLayer, Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.constraints import MaxNorm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def do_my_study():

    df = pd.read_csv('final_project(5).csv')

    df = preproc(df)

    preproc_x, y, X_train, X_test, y_train, y_test = model_prep(df)

    model = do_DenseNeuralNetwork(X_train, y_train, X_test, y_test)
    display_class_report(model, preproc_x, y_test)


def preproc(df):

    # Fix rows of x37
    df['x37'] = df['x37'].str.replace('$', '')

    # Perform median imputations since all numeric vars are normally distributed
    df = df.fillna(df.median())

    df['x24'] = df['x24'].fillna('Unknown')
    df['x29'] = df['x29'].fillna('Unknown')
    df['x30'] = df['x30'].fillna('Unknown')
    df['x32'] = df['x30'].fillna('Unknown')

    return df


def model_prep(df):
    y = df['y']

    categorical_cols = [i for i in df.columns if df.dtypes[i]=='object']
    dummied_cat_cols_df = pd.get_dummies(df[categorical_cols])

    numerical_cols = df._get_numeric_data().columns
    numerical_cols_df = df[numerical_cols].copy().drop('y', axis=1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_num_cols_df = pd.DataFrame(
        scaler.fit_transform(numerical_cols_df), 
        columns=numerical_cols_df.columns.values)
    
    preproc_x = pd.concat([scaled_num_cols_df, dummied_cat_cols_df], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        preproc_x, y, test_size=0.2, random_state=42)

    return preproc_x, y, X_train, X_test, y_train, y_test


def do_DenseNeuralNetwork(X_train, y_train, X_test, y_test):

    model = Sequential()
    model.add(Dense(500, input_shape=(75,), activation='relu', kernel_constraint=MaxNorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(400, activation='relu', kernel_constraint=MaxNorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(300, activation='relu', kernel_constraint=MaxNorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu', kernel_constraint=MaxNorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu', kernel_constraint=MaxNorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    callback = EarlyStopping(monitor='val_loss', patience=3)

    model.fit(
        X_train, y_train,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=[callback],
        batch_size=100)

    return model


def display_class_report(model, preproc_x, y_test):
    y_pred = model.predict(preproc_x)
    y_pred = np.round(y_pred)

    cm = confusion_matrix(y, y_pred)
    TN, FP, FN, TP = cm.ravel()

    print(classification_report(y_test, y_pred))
    print(FN*15 + FP*35)


if __name__ == "__main__":

    do_my_study()
