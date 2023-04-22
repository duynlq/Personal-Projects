# Personal Projects
Collection of data science projects using datasets from Kaggle.

Tools used: 
+ *Python (Pandas, Numpy, Pyplot, Seaborn, scikit-learn, imbalanced-learn, email, nltk, re, scipy, xgboost, Tensorflow)* 
+ *R (tidyverse, DataExplorer, caret, MASS, regclass, ggplot2, glmtoolbox, GGally, lindia, gridExtra, glmnet, ROCR, stats, MLmetrics, mvtnorm, RColorBrewer, pheatmap, cluster, jtools, broom)*

## Contents

#### Machine Learning
+ [Loan-likelihood Prediction](https://github.com/duynlq/Personal-Projects/blob/main/loan_likelihood/loan_likelihood.pdf): Logistic regression model that predicts whether or not a bank customer will accept a loan based on several personal factors. Pushes boundaries of logistic regression by testing various feature selection methods (forward, backward, and stepwise) and feature engineering methods (polynomials and variable interactions). The final model achieved an accuracy of 97% and true positive and true negative rates of 92% and 98%, respectively.
+ [Spam Classifier](https://github.com/duynlq/Personal-Projects/blob/main/spam_classifier/spam_classifier.ipynb): Predicting whether an email is spam using word relevancy and text segmentation. Texts from emails are vectorized into a TF-IDF matrix, and K-means++ clustering is used to create a column of clusters that each email belongs to, which is then appended to the matrix. The final model uses the Gaussian Naive-Bayes classifier and achieves 99% accuracy. Uses a subset of this [data](https://spamassassin.apache.org/old/publiccorpus/).
+ [Bankruptcy Prediction](https://github.com/duynlq/Personal-Projects/blob/main/bankrupty_classifier/bankrupty_classifier.ipynb): Predicting whether a company will go bankrupt using Random Forest (RF) and Extreme Gradiant Boosting (XGBoost). Hypertuning is done by using random search. The mean and standard deviation Area Under the Curve (AUC) is used as the comparison metric, where the RF model achieves 0.846 and 0.034 respectively, and the XGBoost model achieves 0.925 and 0.033 respectively.
+ [Firewall Traffic Detection](https://github.com/duynlq/Personal-Projects/blob/main/firewall_traffic/%20firewall_traffic.ipynb) Multinomial logistic regression model that classifies incoming firewall traffic as allow, drop, deny, or reset-both. Least accessed ports are grouped into one for each port type. Hypertuning is done using GridSearchCV, where C-Support Vector Classification and SGD Classifier are used, achieving 100% and 99% accuracy, respectively. 
+ [Budget Loss Prediction](https://github.com/duynlq/Personal-Projects/blob/main/budget_loss/%20budget_loss.ipynb) Dense neural network on 160,000 rows of anonymous data that focuses on predicting the least false negatives and false positives. With budget loss of $15 assigned for false negatives and $35 for false positives, this network achieves $270,000 lost with 94% accuracy when predicting on the entire preprocessed data.



