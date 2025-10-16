import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#pobieranie datasetów
path = kagglehub.dataset_download("krzysztofjamroz/apartment-prices-in-poland")

#wywalenie wszystkich plikow dotyczacych wynajmów
for file_name in os.listdir(path):
    if "rent" in file_name.lower():
        file_path = os.path.join(path, file_name)
        os.remove(file_path)

#polaczenie wszystkich pozostalych plikow w jeden
files = os.listdir(path)
df_list = [pd.read_csv(os.path.join(path, f)) for f in files]
df = pd.concat(df_list, ignore_index=True)

print(df.head())
df.info()
print(df.isnull().sum())
print(df.describe())
print(df.columns)