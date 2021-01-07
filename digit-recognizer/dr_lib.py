import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC as SVM
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid as NC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

# carrega uma matriz numpy a partir de uma linha do dataset
def load_image(row):
    return np.array(row).reshape(28, 28) / 255

# funcaozinha que gera imagens do dataset soh de brincadeira
def plot_digits(dataset):
    for index, row in dataset.iterrows():
        img = load_image(row)
        plt.imsave("figures/digits/{}.png".format(index), img, vmin=0, vmax=255, cmap="binary")
        print(index, "ok!")

# calcula Momentos de Hu
def extract_hu(dataset, log_transform=False):
    X = []
    
    for index, row in dataset.iterrows():
        img = load_image(row)
        hu = cv2.HuMoments(cv2.moments(img))
        
        if log_transform:
            for i in range(0, 7):
                hu[i] = -1 * np.sign(hu[i]) * np.log10(np.abs(hu[i]))
        
        X.append(np.ravel(hu))
    
    return X

