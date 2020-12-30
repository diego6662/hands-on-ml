import os 
import tarfile
import urllib.request as urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import  train_test_split
download_url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
housing_path = os.path.join("datasets","housing")
housing_url = download_url + "datasets/housing/housing.tgz"
def fetch_housing_data(housing_url=housing_url, housing_path = housing_path):
    os.makedirs(housing_path,exist_ok=True)
    tgz_path =  os.path.join(housing_path,'housing.tgz')
    urllib.urlretrieve(housing_url,tgz_path)
    housin_tgz = tarfile.open(tgz_path)
    housin_tgz.extractall(path=housing_path)
    housin_tgz.close()
#call the fecth function to create dir and download the data
fetch_housing_data()

def load_housing_data(housing_path=housing_path):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

housing_df = load_housing_data()
print(housing_df.head())
input()
print(housing_df.info())
input()
#take a look to the ocean_proximity categories and describe full dataset with describe() method
print(housing_df['ocean_proximity'].value_counts())
input()
print(housing_df.describe())
input()
housing_df.hist(bins=50,figsize=(20,15))
plt.show()

#split the data
def split_train_test(data,test_ratio):
    np.random.seed(12)
    shuffle_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffle_indices[:test_set_size]
    train_indices = shuffle_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

train_set, test_set = train_test_split(housing_df,test_size=0.2,random_state=42)

print(len(train_set),len(test_set))

