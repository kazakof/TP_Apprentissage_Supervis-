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

Features = pd.read_csv("TP2-complementary data/acsincome_ca_features_without_race.csv")  # Features
Label = pd.read_csv("./acsincome_ca_labels.csv")   # labels

# Remplacer les valeurs booléennes par 1 et 0 dans le DataFrame Label
Label = Label.replace({True: 1, False: 0})

# Fusionner Features et Label en utilisant la colonne d'index
data = pd.concat([Features, Label], axis=1)

# Séparer les données en features et label
X_train, _, Y_train, _ = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=42)

# Concaténer les features et le label pour calculer les corrélations
data_train = pd.concat([X_train, Y_train], axis=1)

# Calculer les corrélations
correlations = data_train.corr()

X_all = np.array(Features)
Y_all = np.array(Label).ravel()

X_all, Y_all = shuffle(X_all, Y_all, random_state=1)

# only use the first N samples to limit training time
ratio = 0.01 
print(len(X_all))


num_samples = int(len(X_all)*ratio)
print(num_samples)

X_sampled, Y_sampled = X_all[:num_samples], Y_all[:num_samples]

X_train, X_test,Y_train, Y_test = train_test_split(X_sampled,Y_sampled, test_size=0.2)


#Logistic Regression 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
modelLR2 = LogisticRegression(random_state=0)
modelLR2.fit(X_train_scaled,Y_train)
modelLR2.predict(X_test_scaled)
scores_logisticRegression = modelLR2.score(X_train_scaled,Y_train)


#SVM CLASSIFIER 

modelSVM = make_pipeline(StandardScaler(),SVC(random_state=0))
modelSVM.fit(X_train,Y_train)

# Make predictions
Y_pred_train_svm = modelSVM.predict(X_train)
Y_pred_test_svm = modelSVM.predict(X_test)
score_SVM = sklearn.metrics.accuracy_score(Y_train, Y_pred_train_svm)


# RANDOM FOREST CLASSIFIER

from sklearn.ensemble import RandomForestClassifier

# Create a RandomForest Classifier
modelRF = RandomForestClassifier(random_state=0)

# Train the model
modelRF.fit(X_train, Y_train)

# Make predictions
Y_pred_train_rf = modelRF.predict(X_train)
Y_pred_test_rf = modelRF.predict(X_test)

score_RF = sklearn.metrics.accuracy_score(Y_train, Y_pred_train_rf)

# ADABOOST CLASSIFIER

from sklearn.ensemble import AdaBoostClassifier

# Create an AdaBoost Classifier
modelAdaBoost = AdaBoostClassifier(random_state=0)

# Train the model
modelAdaBoost.fit(X_train, Y_train)

# Make predictions
Y_pred_train_adaboost = modelAdaBoost.predict(X_train)
Y_pred_test_adaboost = modelAdaBoost.predict(X_test)

score_Adaboost = sklearn.metrics.accuracy_score(Y_train, Y_pred_train_adaboost)



# GRADIENT BOOSTING CLASSIFIER

from sklearn.ensemble import GradientBoostingClassifier

# Create a GradientBoosting Classifier
modelGradientBoost = GradientBoostingClassifier(random_state=0)

# Train the model
modelGradientBoost.fit(X_train, Y_train)

# Make predictions
Y_pred_train_gradientboost = modelGradientBoost.predict(X_train)
Y_pred_test_gradientboost = modelGradientBoost.predict(X_test)

score_GB = sklearn.metrics.accuracy_score(Y_train, Y_pred_train_gradientboost)

from sklearn.metrics import confusion_matrix

# List of models
models = [modelLR2, modelSVM, modelRF, modelAdaBoost, modelGradientBoost]
model_names = ['Logistic Regression', 'SVM', 'Random Forest', 'AdaBoost', 'Gradient Boosting']

def calculate_disparate_impact(y_true, y_pred, sensitive_feature_values):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate disparate impact
    favorable_outcome_index = 1  # Assuming 1 represents the favorable outcome
    disparate_impact = (cm[1, 1] / (cm[1, 1] + cm[0, 1])) / (cm[1, 0] / (cm[1, 0] + cm[0, 0]))

    return disparate_impact

def calculate_statistical_parity_difference(y_true, y_pred, sensitive_feature_values):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate statistical parity difference
    favorable_outcome_index = 1  # Assuming 1 represents the favorable outcome
    spd = (cm[1, 0] / (cm[1, 0] + cm[0, 0])) - (cm[1, 1] / (cm[1, 1] + cm[0, 1]))

    return spd


X_train_no_sex = X_train
X_test_no_sex = X_test

# Loop through each model
for i, model in enumerate(models):
    # Remove the 'SEX' feature from the model if it exists
    if hasattr(model, 'named_steps') and 'logisticregression' in model.named_steps:
        model_without_sex = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
    else:
        model_without_sex = model

    # Make predictions on data without 'SEX'
    Y_pred_train_no_sex = model_without_sex.predict(X_train_no_sex)
    Y_pred_test_no_sex = model_without_sex.predict(X_test_no_sex)

    # Calculate metrics for the training set without 'SEX'
    di_train_no_sex = calculate_disparate_impact(Y_train, Y_pred_train_no_sex, None)  # No sensitive feature
    spd_train_no_sex = calculate_statistical_parity_difference(Y_train, Y_pred_train_no_sex, None)  # No sensitive feature

    # Calculate metrics for the test set without 'SEX'
    di_test_no_sex = calculate_disparate_impact(Y_test, Y_pred_test_no_sex, None)  # No sensitive feature
    spd_test_no_sex = calculate_statistical_parity_difference(Y_test, Y_pred_test_no_sex, None)  # No sensitive feature

    # Display results without 'SEX'
    print(f"Model: {model_names[i]} (Without 'SEX')")
    print(f"Disparate Impact (Training Set): {di_train_no_sex}")
    print(f"Statistical Parity Difference (Training Set): {spd_train_no_sex}")
    print("="*50)
    print(f"Disparate Impact (Test Set): {di_test_no_sex}")
    print(f"Statistical Parity Difference (Test Set): {spd_test_no_sex}")
    print("="*50)

