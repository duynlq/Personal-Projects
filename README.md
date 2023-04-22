# Personal Projects
Collection of data science projects using datasets from Kaggle.

Tools used: 
+ *Python (Pandas, Numpy, scikit-learn, imbalanced-learn, email, nltk, re, scipy, xgboost)* 
+ *R (tidyverse, DataExplorer, caret, MASS, regclass, ggplot2, glmtoolbox, GGally, lindia, gridExtra, glmnet, ROCR, stats, MLmetrics, mvtnorm, RColorBrewer, pheatmap, cluster, jtools, broom)*

## Contents

#### Machine Learning
+ [Loan-likelihood Prediction](https://github.com/duynlq/Personal-Projects/blob/main/loan_likelihood/duyProject2.pdf): A classifier that predicts whether or not a bank customer will accept a loan based on several personal factors. Pushes boundaries of logistic regression by testing various feature selection methods (forward, backward, and stepwise) and feature engineering methods (polynomials and variable interactions). The final model achieved an accuracy of 97% and true positive and true negative rates of 92% and 98%, respectively.
+ [Spam Prediction](https://github.com/duynlq/Personal-Projects/blob/main/spam_classifier/Duy_Nguyen_CaseStudy3.ipynb): Predicting whether an email is spam using word relevancy and text segmentation. Texts from emails are vectorized into a TF-IDF matrix, and K-means++ clustering is used to create a column of clusters that each email belongs to, which is then appended to the matrix. The final model uses the Gaussian Naive-Bayes classifier and achieves 99% accuracy. Uses a subset of this [data](https://spamassassin.apache.org/old/publiccorpus/).
+ [Bankruptcy Prediction](https://github.com/duynlq/Personal-Projects/blob/main/bankrupty_classifier/Duy_Nguyen_CaseStudy4.ipynb): Predicting whether a company will go bankrupt using Random Forest (RF) and Extreme Gradiant Boosting (XGBoost). The mean and standard deviation Area Under the Curve (AUC) is used as the comparison metric, where the RF model achieves 0.846 and 0.034 respectively, and the XGBoost model achieves 0.925 and 0.033 respectively.



