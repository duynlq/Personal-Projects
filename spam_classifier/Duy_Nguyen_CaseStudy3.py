import warnings
import sys
from os import listdir
from os.path import isfile, join
import pandas as pd
import email
import nltk
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")


def do_my_study():
    df, msgs = import_data()

    # print(df[df["filename"].str.contains('00947')])
    # print(len(msgs))

    tf_matrix = create_tfidf_matrix(df)

    # print(tf_matrix.shape[0])

    clusters = 3
    df = assign_clusters(df, clusters, tf_matrix, 'kmeans_cluster')

    # print(df['kmeans_cluster'].value_counts())

    tf_matrix_array_cluster = add_column_to_tf_matrix(
        tf_matrix, df, 'kmeans_cluster')

    # print(tf_matrix_array_cluster.shape)

    perform_NB_algorithm_and_evaluate(
        GaussianNB(),
        X=tf_matrix_array_cluster, y=df['is_spam'],
        threshold=0.5)


def import_data():
    user_input_path = sys.argv[1]

    directories = [
            'easy_ham',
            'easy_ham_2',
            'hard_ham',
            'spam',
            'spam_2'
            ]

    msgs = []
    file_name = []
    label = []
    for d in directories:
        mypath = user_input_path + '/SpamAssassinMessages/' + d + '/'
        # mypath = getcwd() + '/SpamAssassinMessages/' + d + '/'
        myfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

        for file in myfiles:
            with open(mypath + file, encoding='latin1') as f:
                msg = email.message_from_file(f)
                for part in msg.walk():
                    if "spam" in d:
                        label.append(1)
                    else:
                        label.append(0)
                    file_name.append(file)
                    payload = part.get_payload()
                    msgs.append(payload)

    df = pd.DataFrame({"filename": file_name, "is_spam": label})
    df['messages'] = msgs

    return df, msgs


def normalize_text_func(text):
    stop_words = nltk.corpus.stopwords.words('english')

    # make all special characters lowercase and remove them
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text, re.I | re.A)
    text = text.lower()
    text = text.strip()

    # tokenize document
    tokens = nltk.word_tokenize(text)

    # filter out stop words
    filtered_tokens = [token for token in tokens
                       if token not in stop_words]

    # Remove numbers
    filtered_tokens = [token for token in filtered_tokens
                       if not token.isdigit()]

    # Remove short tokens
    filtered_tokens = [token for token in filtered_tokens
                       if len(token) > 2]

    # re-create a normalized document
    text = ' '.join(filtered_tokens)
    return text


def create_tfidf_matrix(df):
    normalize_text = np.vectorize(normalize_text_func)
    normalized_text = normalize_text(df['messages'].astype('str'))

    vectorizer = TfidfVectorizer()
    tf_matrix = vectorizer.fit_transform(str(i) for i in normalized_text)

    return tf_matrix


def assign_clusters(df, clusters, tf_matrix, column_name):
    km = KMeans(n_clusters=clusters,
                max_iter=10000,
                n_init=10,
                random_state=42).fit(tf_matrix)
    df[column_name] = km.labels_

    return df


def add_column_to_tf_matrix(tf_matrix, df, column_name):
    tf_matrix_array = tf_matrix.toarray()
    tf_matrix_array_column = np.append(
        tf_matrix_array, df[column_name].values.reshape(-1, 1), axis=1)

    return tf_matrix_array_column


def perform_NB_algorithm_and_evaluate(algorithm, X, y, threshold):
    algorithm.fit(X, y)

    y_preds = (algorithm.predict_proba(X)[:, 1] > threshold).astype('float')

    print("Classification Report ( Threshold = ", threshold,
          ") ( Algorithm = ", algorithm, "):")
    print(classification_report(y, y_preds, digits=4))


if __name__ == "__main__":

    do_my_study()
