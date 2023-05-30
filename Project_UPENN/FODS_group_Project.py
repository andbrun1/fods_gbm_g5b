import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")


data_clinical = pd.read_csv("./clinFeatures_UPENN.csv")
data_radiation = pd.read_csv("./radFeatures_UPENN.csv")

print(data_clinical.shape)
