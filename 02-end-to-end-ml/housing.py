import os 
import tarfile
import urllib.request as urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import  train_test_split, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import  cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy import stats
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

#analize the median income attribute

housing_df['income_cat'] = pd.cut(housing_df['median_income'],
                                  bins=[0.,1.5,3.,4.5,6.,np.inf],
                                  labels=[1,2,3,4,5])
housing_df['income_cat'].hist()
plt.show()
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing_df, housing_df['income_cat']):
    strat_train =  housing_df.loc[train_index]
    strat_test = housing_df.loc[test_index]

print(strat_test['income_cat'].value_counts() / len(strat_test))

for set_ in (strat_train,strat_test):
    set_.drop("income_cat",axis=1,inplace=True)
#EXPLORE THE DATA
#create a copy of the train set
strat_train_copy = strat_train.copy()

#plot geographical data 

strat_train_copy.plot(kind='scatter',x='longitude',y='latitude',alpha=0.4,
        s=strat_train_copy['population']/100,label='population',figsize=(10,7),
        c='median_house_value',cmap=plt.get_cmap("jet"),colorbar=True)
plt.legend()
plt.show()

#looking the correlations between attributes
corr_matrix = strat_train_copy.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))
attributes = ['median_house_value', 'median_income', 'total_rooms',
              'housing_median_age']
scatter_matrix(strat_train_copy[attributes],figsize=(12,8))
plt.show()

#creating new attributes

strat_train_copy['rooms_per_household'] = strat_train_copy['total_rooms'] / strat_train_copy['households']
strat_train_copy['bedrooms_per_room'] = strat_train_copy['total_bedrooms'] / strat_train_copy['total_rooms']
strat_train_copy['population_per_household'] = strat_train_copy['population'] / strat_train_copy['households']
corr_matrix = strat_train_copy.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))
housing = strat_train.drop("median_house_value", axis=1)
housing_labels = strat_train["median_house_value"].copy()

#------------------- DATA CLEANING ------------------#
median = housing['total_bedrooms'].median()
print(housing['total_bedrooms'].isnull().sum())
housing['total_bedrooms'].fillna(median,inplace=True)
print(housing['total_bedrooms'].isnull().sum())

imputer = SimpleImputer(strategy='median')
housing_num = housing.drop('ocean_proximity',axis=1)
imputer.fit(housing_num)

#transfor the data
x = imputer.transform(housing_num)
housing_tr = pd.DataFrame(x, columns = housing_num.columns,
                          index=housing_num.index)
print(housing_tr.head())
#categorical values

house_cat = housing[['ocean_proximity']]
print(house_cat.head())

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(house_cat)
print(housing_cat_encoded[:10])

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(house_cat)
print(housing_cat_1hot.toarray())

num_pipeline = Pipeline([('imputer',SimpleImputer(strategy='median')),
                          ('std_scaler',StandardScaler())])
num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']
full_pipeline = ColumnTransformer([("num",num_pipeline,num_attribs),
                                   ('cat',OneHotEncoder(),cat_attribs)])
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)

#--------TRAINING AND EVALUATING ON THE TRAINING SET------------------#
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data = full_pipeline.transform(some_data)
housing_predict = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels,housing_predict)
print("linear regression mse:",lin_mse)

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predict = tree_reg.predict(housing_prepared)
tree_rmse = mean_squared_error(housing_labels,housing_predict)
tree_rmse = np.sqrt(tree_rmse)
print("decision tree regression rmse:",tree_rmse)
#---------------------------- BETTER EVALUATION USING CROSS-VALIDATION----------#
scores = cross_val_score(tree_reg,housing_prepared,housing_labels,scoring='neg_mean_squared_error',cv=10)
tree_rmse = np.sqrt(- scores)
print("decision tree regression rmse(using cross-validation):",tree_rmse)

lin_scores = cross_val_score(lin_reg,housing_prepared,housing_labels,scoring='neg_mean_squared_error',cv=10)
lin_rmse_score = np.sqrt(-lin_scores)
print('linear regression rmse(using cross-validation):',lin_rmse_score)

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared,housing_labels)
forest_score = cross_val_score(forest_reg,housing_prepared,housing_labels,scoring='neg_mean_squared_error',cv=10)
forest_rmse_score = np.sqrt(-forest_score)
print('random forest rmse(using cross-validation):',forest_rmse_score)

#------------- FINE-TUNE YOUR MODEL -------------------------------------------#
#GRID SEARCH

param_grid = [
        {'n_estimators':[3,10,30],'max_features':[2,4,6,8]},
        {'bootstrap':[False,True],'n_estimators':[3,10],'max_features':[2,3,4]},
        ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg,param_grid,cv=5,
                           scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(housing_prepared,housing_labels)
print(grid_search.best_params_)
input()
print(grid_search.best_estimator_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score),params)

#-------- EVALUATE YOUR SYSTEM ON THE TEST SET --------------------------$
final_model = grid_search.best_estimator_
x_test = strat_test.drop("median_house_value",axis=1)
y_test = strat_test['median_house_value'].copy()

x_test_prepared = full_pipeline.transform(x_test)
final_predictions = final_model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test,final_predictions)
final_rmse = np.sqrt(final_mse)
print('final rmse',final_rmse)
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2

print(np.sqrt(stats.t.interval(confidence,len(squared_errors) - 1,loc=squared_errors.mean(),scale=stats.sem(squared_errors))))
