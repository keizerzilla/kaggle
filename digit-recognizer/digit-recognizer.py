from dr_lib import *

# preparando ambiente de pastas
os.makedirs("results/", exist_ok=True)
os.makedirs("figures/digits/", exist_ok=True)

# carregando conjuntos de dados
train = pd.read_csv("train.csv")
y = train["label"]
train = train.drop(["label"], axis=1)
test = pd.read_csv("test.csv")

# classificadores
classifiers = {
    "NC"         : NC(),
    "LDA"        : LDA(),
    "QDA"        : QDA(),
    "SVM_linear" : SVM(kernel="linear"),
    "SVM_radial" : SVM(kernel="rbf")
}

print("Extraindo momentos do conjunto de treinamento...")
X = extract_hu(train, True)
#print("Extraindo momentos do conjunto de teste...")
#X_test = extract_hu(test)
print("Ok!")

test_ids = list(range(1, test.shape[0]+1))

for name, clf in classifiers.items():
    for r in range(100):
        # criando conjuntos de validacao por eu jah ter o label do conjunto de treinamento
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        pt = PowerTransformer()
        pt.fit(X_train)
        X_train = pt.transform(X_train)
        X_test = pt.transform(X_test)
        
        clf.fit(X_train, y_train)
        score = round(clf.score(X_test, y_test), 2)
        
        print("round: {}\tclf: {}\tscore: {}".format(r, name, score))
    
    # abaixo o codigo que faz predicoes a partir do conjunto de treinamento completo
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    df = pd.DataFrame(data={"ImageId" : test_ids, "Label" : y_pred})
    df.to_csv("results/{}.csv".format(name), index=None)
    print(name, "ok!")
    """
