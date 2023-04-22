# Personal Projects
Collection of data science projects using datasets from Kaggle.

*Tools used: Python (Pandas, Numpy, scikit-learn, email, nltk, re), R (tidyverse, DataExplorer, caret, MASS, regclass, ggplot2, glmtoolbox, GGally, lindia, gridExtra, glmnet, ROCR, stats, MLmetrics, mvtnorm, RColorBrewer, pheatmap, cluster, jtools, broom)*

## Contents

#### Machine Learning
+ [Spam Classifier](https://github.com/duynlq/Personal-Projects/blob/main/spam_classifier/Duy_Nguyen_CaseStudy3.ipynb): Predicting whether an email is spam by using word relevancy and text segmentation. Texts from emails are vectorized into a TF-IDF matrix, and K-means++ clustering is used to create a column of clusters that each email belongs to, which is then appended to the matrix. The final model uses the Gaussian Naive-Bayes classifier and achieves 99% accuracy. Uses a subset of this [data](https://spamassassin.apache.org/old/publiccorpus/).

+ [Loan-likelihood Prediction](https://github.com/duynlq/Personal-Projects/blob/main/loan_likelihood/duyProject2.pdf): A classifier that predicts whether or not a bank customer will accept a loan based on several personal factors. Pushes boundaries of logistic regression by testing various feature selection methods (forward, backward, and stepwise) and feature engineering methods (polynomials and variable interactions). The final model achieved an accuracy of 97% and true positive and true negative rates of 92% and 98%, respectively.

