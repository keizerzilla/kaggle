"""
classifier.py
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from joblib import dump
from joblib import load

def calc_performance_metrics(y_test, y_pred, description="Metrics"):
    """ Função auxiliar que calcula métricas de performance do modelo:
    1- Revocação;
    2- Precisão;
    3- F1;
    4- Acurácia;
    5- Matriz de confusão.
    
    Parâmetros
    ----------
    y_test: vetor com etiquetas reais das amostras.
    y_pred: vetor com as etiquetas preditas pelo classificador.
    description: uma descrição do experimento para ser adicionada ao dataframe de métricas.
    
    Retornos
    --------
    cm: matriz de confusão (métrica 5).
    metrics: dataframe com as métricas de 1 a 4.
    """
    
    cm = confusion_matrix(y_test, y_pred)
    model_recall = recall_score(y_test, y_pred)
    model_precision = precision_score(y_test, y_pred)
    model_f1 = f1_score(y_test, y_pred)
    model_accuracy = accuracy_score(y_test, y_pred)
    
    mscores = [(description, model_recall, model_precision, model_f1, model_accuracy)]
    mcols = ["Description", "Recall", "Precision", "F1_score", "Accuracy"]
    metrics = pd.DataFrame(data=mscores, columns=mcols)
    
    return cm, metrics

class Classifier:
    
    def __init__(self):
        """ Construtor da classe Classifier.
        """
        
        self.X = None
        self.y = None
        self.best_params = None
        self.best_cv_score = None
        self.model = None
        self.cm = None
        self.metrics = None
    
    def from_csv(self, csv_file, in_cols, out_cols):
        """ Define vetores de entrada e saída a partir de arquivo CSV.
        
        Parâmetros
        ----------
        csv_file: caminho completo para o arquivo CSV.
        in_cols: lista com os nomes das colunas a serem usadas como entradas do modelo.
        out_cols: lista com os nomes das colunas a serem usadas como saídas do modelo.
        """
        
        df = pd.read_csv(csv_file)
        self.X = np.array(df[in_cols])
        self.y = np.array(df[out_cols]).ravel()
    
    def from_vectors(self, X, y):
        """ Define vetores de entrada e saída a partir de arranjos pré-calculados.
        
        Parâmetros
        ----------
        X: vetor com variáveis de entrada.
        y: vetor com variáveis de saída.
        """
        
        self.X = X
        self.y = y
    
    def fit_best_parameters(self, pipe, search_space, test_size=0.3, n_splits=5, n_jobs=8, verbose=3, scoring="recall"):
        """ Gera o melhor modelo de classificação usando uma estratégia de validação cruzada
        aliada com otimização de hiperparâmetros a partir de busca exaustiva.
        
        Parâmetros
        ----------
        pipe: pipeline de classificação a ser calibrado.
        search_space: o conjunto de possibilidades para o hiperparâmetros a serem testados.
        test_size: o tamanho da parcela de dados dedicada para teste.
        n_splits: número de conjuntos de validação cruzada a ser usado.
        n_jobs: quantos processadores usar durante a geração do modelo (paralelização).
        verbose: nível de debug durante calibração dos hiperparâmetros.
        scoring: a métrica a ser otimizada: revocação ("recall") ou precisão ("precision").
        """
        
        if self.X is None or self.y is None:
            raise Exception("One or both data vectors are empty!")
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, stratify=self.y, test_size=test_size)
        kf = KFold(n_splits=n_splits)
        
        grid = GridSearchCV(pipe, param_grid=search_space, cv=kf, scoring=scoring, verbose=verbose, n_jobs=n_jobs)
        grid = grid.fit(X_train, y_train)
        
        y_pred = grid.predict(X_test)
        metrics, cm = calc_performance_metrics(y_test, y_pred)
        
        self.best_params = grid.best_params_
        self.best_cv_score = grid.best_score_
        self.model = grid.best_estimator_
        self.cm = cm
        self.metrics = metrics
        
        if verbose > 0:
            print(f"Best params: {self.best_params}")
            print(f"Best score: {self.best_cv_score}")
            print(self.cm)
            print(self.metrics)
            print(self.model)
    
    def fit_best_classifier(self):
        """ Gera o melhor classificador treinando com todos os dados disponíveis.
        """
        
        if self.X is None or self.y is None:
            raise Exception("One or both data vectors are empty!")
        
        self.model = self.model.fit(self.X, self.y)
    
    def create_submission(self, X_test, sample_submission, id_name, id_col, sub_file):
        """ Executa o modelo em dados de teste e formata a predição no formato esperado pela competição.
        
        Parâmetros
        ----------
        X_test: matriz com dados de teste.
        sample_submission: exemplo de arquivo CSV contendo o formato de submissão.
        id_name: nome da coluna de identificação das predições.
        id_col: lista com os ids a serem posicionados à esquerda da submissão de predições.
        sub_file: caminho para um arquivo CSV contendo a submissão criada a partir da predição.
        
        Retornos
        --------
        sub: DataFrame contendo a submissão que foi gerada e salva.
        """
        
        if self.model is None:
            raise Exception("The model was not calibrated yet!")
        
        y_pred = self.model.predict(X_test).tolist()
        
        sample_df = pd.read_csv(sample_submission)
        sub_cols = sample_df.columns.tolist()
        output_col = list(set(sub_cols) - set(id_name))[0]
        
        sub = pd.DataFrame(columns=sub_cols)
        sub[id_name] = id_col
        sub[output_col] = y_pred
        
        sub.to_csv(sub_file, index=None)
        
        return sub
    
    def save_model(self, model_file):
        """ Salva modelo gerado em arquivo para uso a posteriori.
        
        Parâmetros
        ----------
        model_file: caminho para o arquivo com o modelo salvo em disco.
        """
        
        if self.model is None:
            raise Exception("The model was not calibrated yet!")
        
        dump(self.model, model_file)
    
    def load_model(self, model_file):
        """ Carrega modelo de um arquivo para uso.
        
        Parâmetros
        ----------
        model_file: caminho para o arquivo com o modelo salvo em disco.
        """
        
        self.model = load(model_file)
    