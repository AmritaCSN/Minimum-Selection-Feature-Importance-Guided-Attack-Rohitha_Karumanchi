import warnings
warnings.filterwarnings("ignore")
from src import figa_MS as figa
import pandas as pd
import numpy as np
import csv
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("../data/FIGA_data.csv")

def select_target(data):
    #take phishing samples
    test_data=  data[ data["label"]==1]
    
    #select the targetted phishing sample and store it.
    X_target= test_data.sample(1) #select a phishing sample
    X_target=X_target.drop(columns="label")
    y_target= pd.Series([1], index=X_target.index)

    #Delete the target sample from data
    data=data.drop(X_target.index)
    print(X_target.index)

    X = data.drop("label", axis=1)
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    print(f'{X_train.isna().any().any()}')
    return (X_target,y_target,X_train,y_train)

#XG Boost
xgb_model=xgb.XGBClassifier()
scaler = MinMaxScaler(feature_range=(0, 1))
model = make_pipeline(scaler,xgb_model)

xgb_result=np.array([[0] for i in range(2)])# 2d array with 100 rows and 1 column
#print(type(xgb_result))
with open("xgb_data.csv","a") as f:
    data_writer = csv.writer(f)
    for i in range(0,2):
            print(f"XGB Iteration : {i}")
            X_target,y_target,X_train,y_train=select_target(data)
            xgb_target_model = model.fit(X_train, y_train)
            xgb_attack=figa.MS_FIGA(X_train, y_train, "rfe")
            sample=xgb_attack.generate(xgb_target_model,X_target,y_target,e=4.0,n=20)
            xgb_result[i][0] = sample
            sample_data=xgb_result.flatten()
            data_writer.writerow(sample_data)
xgb_list = xgb_result.flatten()
xgb_min_sample = sum(xgb_list) / len(xgb_list)
print(f" xgb result {xgb_min_sample}  and xgb samples {xgb_result}")
# open CSV file in append mode
with open('xgb_attack.csv', mode='a', newline='') as file:
    # create CSV writer object
    writer = csv.writer(file)
    # write each row of data to CSV file
    for row in xgb_result:
        writer.writerow(row)
