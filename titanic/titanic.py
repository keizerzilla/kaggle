import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# preparando ambiente
os.makedirs("results/", exist_ok=True)
os.makedirs("figures/", exist_ok=True)

# carregando datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = train["Survived"]
test_ids = test["PassengerId"]

# descrevendo conjuntos
print(train.head())
print(train.describe())

# limpeza básica das colunas
train = train.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
test = test.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# codificando os atributos textuais
for encode in ["Sex", "Embarked"]:
    le = LabelEncoder()
    le.fit(train[encode])
    train[encode] = le.transform(train[encode])
    test[encode] = le.transform(test[encode])

# treinando regressor para prever idades que faltam
train_with_age = train[train["Age"].isnull() == False]
X_age = train_with_age.drop(["Age"], axis=1)
y_age = train_with_age["Age"]

reg = LinearRegression(positive=True)
reg.fit(X_age, y_age)

def predict_age(row):
    if pd.isnull(row["Age"]):
        sample = np.array(row.drop(["Age"])).reshape(1, -1)
        row["Age"] = reg.predict(sample)[0]
    return row

# aplicando regressor para calcular as idades que faltam
train = train.apply(predict_age, axis=1)
test = test.apply(predict_age, axis=1)

# limpando conjunto test
test["Fare"] = test["Fare"].fillna(0)

# mostrando correlação entre os atributos
plt.figure(figsize=(10, 8))
sns.set_theme()
sns.heatmap(train.corr(), annot=True, cmap="RdYlGn")
plt.title("Correlação entre as variáveis: treinamento")
plt.tight_layout()
plt.savefig("figures/correlacao-treinamento.png")
plt.close()

plt.figure(figsize=(10, 8))
sns.set_theme()
sns.heatmap(test.corr(), annot=True, cmap="RdYlGn")
plt.title("Correlação entre as variáveis: teste")
plt.tight_layout()
plt.savefig("figures/correlacao-test.png")
plt.close()

# classificadores
classifiers = {
    "KNN_1_manhattan" : KNeighborsClassifier(n_neighbors=1, p=1),
    "KNN_2_manhattan" : KNeighborsClassifier(n_neighbors=2, p=1),
    "KNN_3_manhattan" : KNeighborsClassifier(n_neighbors=3, p=1),
    "KNN_4_manhattan" : KNeighborsClassifier(n_neighbors=4, p=1),
    
    "KNN_1_euclidean" : KNeighborsClassifier(n_neighbors=1, p=2),
    "KNN_2_euclidean" : KNeighborsClassifier(n_neighbors=2, p=2),
    "KNN_3_euclidean" : KNeighborsClassifier(n_neighbors=3, p=2),
    "KNN_4_euclidean" : KNeighborsClassifier(n_neighbors=4, p=2),
    
    "RandomForest_1000" : RandomForestClassifier(n_estimators=1000),
    
    "AdaBoost_500" : AdaBoostClassifier(n_estimators=500, random_state=1912),
}

X = train

for name, clf in classifiers.items():
    clf.fit(X, y)
    y_pred = clf.predict(test)
    df = pd.DataFrame(data={"PassengerId" : test_ids, "Survived" : y_pred})
    df.to_csv("results/{}.csv".format(name), index=None)

