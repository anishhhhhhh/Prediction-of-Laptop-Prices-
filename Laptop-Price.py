import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRFRegressor

laptop_path='C:\\Users\\An!sh\\Downloads\\new_egg_gaming_laptops.csv'
laptop_data=pd.read_csv(laptop_path)
laptop_data=laptop_data.drop('laptop_name',axis=1)

#missing_values=laptop_data.isnull().sum()
#print(missing_values)


X=laptop_data[['brand_name','touch','display','processor','gpu','ssd']]

#Replacing values in X
X=X.replace(to_replace=['15.6â€','17.3â€','14.0â€'],value='15.6"')

y=laptop_data[['price']]

#from above missing values we can see that there is a missing value in our feature ie price
#Filling that missing values with mean using Imputation technique
my_imputer=SimpleImputer(strategy='mean')
imputed_y=pd.DataFrame(my_imputer.fit_transform(laptop_data[['price']]))
imputed_y.columns=laptop_data[['price']].columns
#print(imputed_y)



#Cheking if any missing values are still present...

#missing_values=imputed_y.isnull().sum()
#print(missing_values)



# splitting X and Y into train and test set
X_train, X_test, imputed_y_train, imputed_y_test=train_test_split(X, imputed_y, test_size=0.2, random_state=0)


#Selection of categorical columns
categorical_columns=[i for i in X_train.columns if 
                     X_train[i].dtype == 'object']

#Selection of numerical columns
numerical_columns=[i for i in X_train.columns if
                   X_train[i].dtype in ['int64','float64']]



#Selecting which technique should be used for missing values in numerical column
#below transformer replaces all NaN values in numreical columns with mode/most_frequent
numerical_transformer=SimpleImputer(strategy='most_frequent')

#Selecting which technique should be used for missing values in categorical column
#below transformer replaces all NaN values in categorical columns with mode/most_frequent and uses OneHotEncoder
categorical_transformer=Pipeline(steps=[('imputation',SimpleImputer(strategy='most_frequent')),
                                        ('OHE',OneHotEncoder(handle_unknown='ignore'))])

#bundling above two transformers to simplify the code
preprocessor = ColumnTransformer(transformers=[('num',numerical_transformer,numerical_columns),
                                               ('cat',categorical_transformer,categorical_columns)])





               
#Note that above transformer only indicate the pipeline to perform these technique.The actual replacing and stuff happens in pipeline 'pp'  





                   
#Define your model
my_model=RandomForestRegressor(n_estimators=1000,random_state=0)

#for boosting
model=XGBRFRegressor(n_estimators=1000,learning_rate=0.05)


#Bundle preprocessing and modeling code in a pipeline
pp=Pipeline(steps=[('preprocessor',preprocessor),
                   ('model',my_model)])

#fit you model
pp.fit(X_train,imputed_y_train.values.ravel())

#Predict your data
predictions=pp.predict(X_test)

print(predictions)

#Checking accuracy
print('MAE: ',mean_absolute_error(predictions, imputed_y_test))
print(X_test.head(10))










