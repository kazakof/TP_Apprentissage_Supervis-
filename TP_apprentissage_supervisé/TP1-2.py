import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Charger les données de la Californie
Features_ca = pd.read_csv("./acsincome_ca_features.csv")  # Features
Label_ca = pd.read_csv("./acsincome_ca_labels.csv")   # labels


# Diviser les données en ensembles d'entraînement et de test pour la Californie
X_train_ca, X_test_ca, Y_train_ca, Y_test_ca = train_test_split(Features_ca, Label_ca, test_size=0.2, random_state=0)

# Initialiser le modèle de régression logistique
modelLR_ca = LogisticRegression(max_iter=1000, random_state=0)

# Entraîner le modèle
modelLR_ca.fit(X_train_ca, Y_train_ca)

# Faire des prédictions sur l'ensemble de test de la Californie
Y_pred_test_ca = modelLR_ca.predict(X_test_ca)

# Afficher la précision du modèle pour la Californie
print(f'Model: Logistic Regression - Train Accuracy (CA): {accuracy_score(Y_train_ca, modelLR_ca.predict(X_train_ca))}')
print(f'Model: Logistic Regression - Test Accuracy (CA): {accuracy_score(Y_test_ca, Y_pred_test_ca)}')
print('========================================')

# Charger les données du Colorado
Features_co = pd.read_csv("./TP2-complementary data/acsincome_co_allfeaturesTP2.csv")
label_co = pd.read_csv("./TP2-complementary data/acsincome_co_labelTP2.csv")

# Charger les données du Nevada
Features_ne = pd.read_csv("./TP2-complementary data/acsincome_ne_allfeaturesTP2.csv")
label_ne = pd.read_csv("./TP2-complementary data/acsincome_ne_labelTP2.csv")

# Diviser les données en ensembles d'entraînement et de test pour le Colorado et le Nevada
X_train_co, X_test_co, Y_train_co, Y_test_co = train_test_split(Features_co, label_co, test_size=0.2, random_state=0)
X_train_ne, X_test_ne, Y_train_ne, Y_test_ne = train_test_split(Features_ne, label_ne, test_size=0.2, random_state=0)

# Appliquer le modèle de régression logistique à l'état du Colorado
Y_pred_test_co = modelLR_ca.predict(X_test_co)
print(f'Model: Logistic Regression - Test Accuracy (CO): {accuracy_score(Y_test_co, Y_pred_test_co)}')
print('========================================')

# Appliquer le modèle de régression logistique à l'état du Nevada
Y_pred_test_ne = modelLR_ca.predict(X_test_ne)
print(f'Model: Logistic Regression - Test Accuracy (NE): {accuracy_score(Y_test_ne, Y_pred_test_ne)}')
print('========================================')
