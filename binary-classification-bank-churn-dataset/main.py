import numpy as np
import pandas as pd

from classifier import Classifier

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import PowerTransformer

def playground_s4e1(sub_id):
    # treinamento
    train_file = "pgs4e1/train.csv"
    df = pd.read_csv(train_file)
    
    # engenharia de atributos
    df["Age_Category"] = pd.cut(df["Age"], bins=[18, 30, 40, 50, 60, 100], labels=["18-30", "30-40", "40-50", "50-60", "60+"], include_lowest=True)
    df["Credit_Score_Range"] = pd.cut(df["CreditScore"], bins=[0, 300, 600, 700, 800, 900], labels=["0-300", "300-600", "600-700", "700-800", "900+"])
    df["Geo_Gender"] = df["Geography"] + "_" + df["Gender"]
    df["Age_Gender"] = df["Age_Category"].astype(str) + "_" + df["Gender"]
    df["HasBalance"] = (df["Balance"] > 0).astype(int)
    
    to_encode = ["Geography", "Gender", "Age_Category", "Credit_Score_Range", "Geo_Gender", "Age_Gender"]
    encoded = [f"{c}_Encoded" for c in to_encode]
    
    enc = OrdinalEncoder()
    enc = enc.fit(df[to_encode])
    
    df[encoded] = enc.transform(df[to_encode])
    
    to_drop = ["id", "CustomerId", "Surname"] + to_encode
    df = df.drop(to_drop, axis=1)
    
    all_cols = list(df.columns)
    output_cols = ["Exited"]
    input_cols = list(set(all_cols) - set(output_cols))
    
    X = np.array(df[input_cols])
    y = np.array(df[output_cols]).ravel()
    
    # pipeline
    scaler = StandardScaler()
    pt = PowerTransformer()
    pca = PCA()
    rf = RandomForestClassifier()
    
    steps = [
        ("scaler", scaler),
        ("pt", pt),
        ("pca", pca),
        ("rf", rf),
    ]
    
    pipe = Pipeline(steps)
    
    search_space = {
        "pca__n_components" : [int(X.shape[1])],
        "rf__n_estimators" : [200, 250, 300, 400],
        "rf__min_samples_leaf" : [2, 4, 6],
    }
    
    clf = Classifier()
    clf.from_vectors(X, y)
    clf.fit_best_parameters(pipe, search_space, n_splits=10, scoring="roc_auc", verbose=3, n_jobs=12)
    
    # teste e submissão
    test_file = "pgs4e1/test.csv"
    id_name = "id"
    sample_submission = "pgs4e1/sample_submission.csv"
    sub_file = f"pgs4e1/my_sub_{sub_id}.csv"
    
    test = pd.read_csv(test_file)
    
    id_col = test[id_name].tolist()
    
    test["Age_Category"] = pd.cut(test["Age"], bins=[18, 30, 40, 50, 60, 100], labels=["18-30", "30-40", "40-50", "50-60", "60+"], include_lowest=True)
    test["Credit_Score_Range"] = pd.cut(test["CreditScore"], bins=[0, 300, 600, 700, 800, 900], labels=["0-300", "300-600", "600-700", "700-800", "900+"])
    test["Geo_Gender"] = test["Geography"] + "_" + test["Gender"]
    test["Age_Gender"] = test["Age_Category"].astype(str) + "_" + test["Gender"]
    test["HasBalance"] = (test["Balance"] > 0).astype(int)
    
    test[encoded] = enc.transform(test[to_encode])
    test = test.drop(to_drop, axis=1)
    
    X_test = np.array(test[input_cols])
    
    clf.fit_best_classifier()
    clf.create_submission(X_test, sample_submission, id_name, id_col, sub_file)
    clf.save_model(f"sample_model_{sub_id}.joblib")
    
if __name__ == "__main__":
	playground_s4e1(1)
	