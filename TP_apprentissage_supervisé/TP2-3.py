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

# Visualiser les corrélations sous forme de heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Corrélations entre features et label')
plt.show()

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