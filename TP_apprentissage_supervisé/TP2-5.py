import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn
import datetime
import calendar
import warnings
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot

Features_without_sex = pd.read_csv("acsincome_ca_features.csv")  # Features
Label = pd.read_csv("./acsincome_ca_labels.csv")   # labels

# Remplacer les valeurs booléennes par 1 et 0 dans le DataFrame Label
Label = Label.replace({True: 1, False: 0})

# Séparer les données en features et label
X_train_no_sex, _, Y_train_no_sex, _ = train_test_split(Features_without_sex, Label, test_size=0.2, random_state=42)

# Répéter le processus de division en ensembles d'entraînement et de test pour les nouvelles données
X_train_no_sex, X_test_no_sex, Y_train_no_sex, Y_test_no_sex = train_test_split(X_train_no_sex, Y_train_no_sex, test_size=0.2, random_state=42)

# Réentraînez vos modèles avec les nouvelles données sans la feature 'SEX'
# Logistic Regression
scaler_no_sex = StandardScaler()
X_train_scaled_no_sex = scaler_no_sex.fit_transform(X_train_no_sex)
X_test_scaled_no_sex = scaler_no_sex.transform(X_test_no_sex)
modelLR2_no_sex = LogisticRegression(random_state=0)
modelLR2_no_sex.fit(X_train_scaled_no_sex, Y_train_no_sex)
scores_logisticRegression_no_sex = modelLR2_no_sex.score(X_train_scaled_no_sex, Y_train_no_sex)

# SVM Classifier
modelSVM_no_sex = make_pipeline(StandardScaler(), SVC(random_state=0))
modelSVM_no_sex.fit(X_train_no_sex, Y_train_no_sex)
score_SVM_no_sex = sklearn.metrics.accuracy_score(Y_train_no_sex, modelSVM_no_sex.predict(X_train_no_sex))

# Random Forest Classifier
modelRF_no_sex = RandomForestClassifier(random_state=0)
modelRF_no_sex.fit(X_train_no_sex, Y_train_no_sex)
score_RF_no_sex = sklearn.metrics.accuracy_score(Y_train_no_sex, modelRF_no_sex.predict(X_train_no_sex))

# AdaBoost Classifier
modelAdaBoost_no_sex = AdaBoostClassifier(random_state=0)
modelAdaBoost_no_sex.fit(X_train_no_sex, Y_train_no_sex)
score_Adaboost_no_sex = sklearn.metrics.accuracy_score(Y_train_no_sex, modelAdaBoost_no_sex.predict(X_train_no_sex))

# Gradient Boosting Classifier
modelGradientBoost_no_sex = GradientBoostingClassifier(random_state=0)
modelGradientBoost_no_sex.fit(X_train_no_sex, Y_train_no_sex)
score_GB_no_sex = sklearn.metrics.accuracy_score(Y_train_no_sex, modelGradientBoost_no_sex.predict(X_train_no_sex))

# Affichez les scores ou toute autre information que vous souhaitez
print("Scores sans la feature 'SEX':")
print("Logistic Regression:", scores_logisticRegression_no_sex)
print("SVM Classifier:", score_SVM_no_sex)
print("Random Forest Classifier:", score_RF_no_sex)
print("AdaBoost Classifier:", score_Adaboost_no_sex)
print("Gradient Boosting Classifier:", score_GB_no_sex)