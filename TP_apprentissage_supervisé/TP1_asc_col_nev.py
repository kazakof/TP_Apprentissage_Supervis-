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

Features = pd.read_csv("./acsincome_ca_features.csv")  # Features
Label = pd.read_csv("./acsincome_ca_labels.csv")   # labels

Label = Label.replace({True: 1, False: 0})

#print(Features)
#print(Label)
label_feature = pd.concat([Features, Label], axis=1)

# Afficher les 10 premières lignes du DataFrame fusionné


selected_values = [8, 32]
Features_colorado_nevada = label_feature[label_feature['POBP'].isin(selected_values)]



# Sélectionner les 10 premières colonnes comme X (caractéristiques)
X_feature_col_nev = Features_colorado_nevada.iloc[:, :-1]

# Sélectionner la dernière colonne comme y (étiquettes)
y_feature_col_nev = Features_colorado_nevada.iloc[:, -1]

print("Features :\n", X_feature_col_nev.head(10))
print("\nLabel :\n", y_feature_col_nev.head(10))

X_all = np.array(X_feature_col_nev)
Y_all = np.array(y_feature_col_nev).ravel()

X_all, Y_all = shuffle(X_all, Y_all, random_state=1)

# only use the first N samples to limit training time
ratio = 1 
print(len(X_all))


num_samples = int(len(X_all)*ratio)
print(num_samples)

X_sampled, Y_sampled = X_all[:num_samples], Y_all[:num_samples]

X_train, X_test,Y_train, Y_test = train_test_split(X_sampled,Y_sampled, test_size=0.2)

#pour afficher les résultats 
results, names = list(), list()




#Logistic Regression 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
modelLR2 = LogisticRegression(random_state=0)
modelLR2.fit(X_train_scaled,Y_train)
modelLR2.predict(X_test_scaled)
scores_logisticRegression = modelLR2.score(X_train_scaled,Y_train)
print('train score LR :', scores_logisticRegression)
print('test score LR :', modelLR2.score(X_test_scaled,Y_test))

results.append(scores_logisticRegression)
names.append('Logistic Regression')

#SVM CLASSIFIER 

modelSVM = make_pipeline(StandardScaler(),SVC(random_state=0))
modelSVM.fit(X_train,Y_train)

# Make predictions
Y_pred_train_svm = modelSVM.predict(X_train)
Y_pred_test_svm = modelSVM.predict(X_test)
score_SVM = sklearn.metrics.accuracy_score(Y_train, Y_pred_train_svm)

# Evaluate the model
print('Train Accuracy SVM:', score_SVM )
print('Test Accuracy SVM:', sklearn.metrics.accuracy_score(Y_test, Y_pred_test_svm))
print('Classification Report SVM:\n', sklearn.metrics.classification_report(Y_test, Y_pred_test_svm))
print('Confusion Matrix SVM:\n', sklearn.metrics.confusion_matrix(Y_test, Y_pred_test_svm))


results.append(score_SVM)
names.append('SVM')

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

# Evaluate the model
print('Train Accuracy RF:', score_RF)
print('Test Accuracy RF:', sklearn.metrics.accuracy_score(Y_test, Y_pred_test_rf))
print('Classification Report RF:\n', sklearn.metrics.classification_report(Y_test, Y_pred_test_rf))
print('Confusion Matrix RF:\n', sklearn.metrics.confusion_matrix(Y_test, Y_pred_test_rf))

results.append(score_RF)
names.append('RF')

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

# Evaluate the model
print('Train Accuracy AdaBoost:', score_Adaboost)
print('Test Accuracy AdaBoost:', sklearn.metrics.accuracy_score(Y_test, Y_pred_test_adaboost))
print('Classification Report AdaBoost:\n', sklearn.metrics.classification_report(Y_test, Y_pred_test_adaboost))
print('Confusion Matrix AdaBoost:\n', sklearn.metrics.confusion_matrix(Y_test, Y_pred_test_adaboost))

results.append(score_Adaboost)
names.append('Ada_boost')


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

# Evaluate the model
print('Train Accuracy GradientBoost:', score_GB)
print('Test Accuracy GradientBoost:', sklearn.metrics.accuracy_score(Y_test, Y_pred_test_gradientboost))
print('Classification Report GradientBoost:\n', sklearn.metrics.classification_report(Y_test, Y_pred_test_gradientboost))
print('Confusion Matrix GradientBoost:\n', sklearn.metrics.confusion_matrix(Y_test, Y_pred_test_gradientboost))

results.append(score_GB)
names.append('GB')

print("labels = ", names)
print("results = ", results)

# Tracer le graphique
plt.figure(figsize=(10, 6))
plt.bar(names, results, color='skyblue')
plt.xlabel('Modèles')
plt.ylabel('Précision')
plt.title('Précision des modèles')

plt.show()


#Définir les hyperparamètres à rechercher pour la régression logistique
param_grid_LR = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

# Créer le GridSearchCV pour la régression logistique
grid_LR = GridSearchCV(LogisticRegression(random_state=0), param_grid_LR, cv=5, scoring='accuracy')
grid_LR.fit(X_train_scaled, Y_train)

# Afficher les meilleurs paramètres et la meilleure précision
print("Meilleurs paramètres pour la régression logistique:", grid_LR.best_params_)
print("Précision avec les meilleurs paramètres:", grid_LR.best_score_)


# Définir les hyperparamètres à rechercher pour SVM
param_grid_SVM = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01, 0.001]}

# Créer le GridSearchCV pour SVM
grid_SVM = GridSearchCV(SVC(random_state=0), param_grid_SVM, cv=5, scoring='accuracy')
grid_SVM.fit(X_train_scaled, Y_train)

# Afficher les meilleurs paramètres et la meilleure précision
print("Meilleurs paramètres pour SVM:", grid_SVM.best_params_)
print("Précision avec les meilleurs paramètres:", grid_SVM.best_score_)


# Définir les hyperparamètres à rechercher pour Random Forest
param_grid_RF = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30]}

# Créer le GridSearchCV pour Random Forest
grid_RF = GridSearchCV(RandomForestClassifier(random_state=0), param_grid_RF, cv=5, scoring='accuracy')
grid_RF.fit(X_train, Y_train)

# Afficher les meilleurs paramètres et la meilleure précision
print("Meilleurs paramètres pour Random Forest:", grid_RF.best_params_)
print("Précision avec les meilleurs paramètres:", grid_RF.best_score_)


# Définir les hyperparamètres à rechercher pour AdaBoost
param_grid_AdaBoost = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}

# Créer le GridSearchCV pour AdaBoost
grid_AdaBoost = GridSearchCV(AdaBoostClassifier(random_state=0), param_grid_AdaBoost, cv=5, scoring='accuracy')
grid_AdaBoost.fit(X_train, Y_train)

# Afficher les meilleurs paramètres et la meilleure précision
print("Meilleurs paramètres pour AdaBoost:", grid_AdaBoost.best_params_)
print("Précision avec les meilleurs paramètres:", grid_AdaBoost.best_score_)

# Définir les hyperparamètres à rechercher pour Gradient Boosting
param_grid_GradientBoost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Créer le GridSearchCV pour Gradient Boosting
grid_GradientBoost = GridSearchCV(GradientBoostingClassifier(random_state=0), param_grid_GradientBoost, cv=5, scoring='accuracy')
grid_GradientBoost.fit(X_train, Y_train)

# Afficher les meilleurs paramètres et la meilleure précision
print("Meilleurs paramètres pour Gradient Boosting:", grid_GradientBoost.best_params_)
print("Précision avec les meilleurs paramètres:", grid_GradientBoost.best_score_)


