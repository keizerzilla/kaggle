import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X = train.drop(["id", "target"], axis=1)
y = train["target"]

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

ptrans = PowerTransformer(standardize=False)
ptrans.fit(X)
X = ptrans.transform(X)

sns.set()

for j in range(X.shape[1]):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    
    ax[0].hist(X[:,j], bins=20)
    sns.regplot(x=X[:,j], y=y, ax=ax[1])
    
    plt.title("feat{}".format(j))
    plt.tight_layout()
    plt.savefig("transformed/{}.png".format(j))
    plt.close()
    
    print(j, "ok")
