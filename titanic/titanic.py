import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier

# carregando datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = train["Survived"]
test_ids = test["PassengerId"]

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

# classificadores simples
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

X = train

eclf = VotingClassifier(estimators=list(zip(names, classifiers)), voting="hard")
eclf.fit(X, y)
y_pred = eclf.predict(test)
df = pd.DataFrame(data={"PassengerId" : test_ids, "Survived" : y_pred})
df.to_csv("eclf.csv", index=None)

for name, clf in zip(names, classifiers):
    clf.fit(X, y)
    y_pred = clf.predict(test)
    df = pd.DataFrame(data={"PassengerId" : test_ids, "Survived" : y_pred})
    df.to_csv("{}.csv".format(name), index=None)

