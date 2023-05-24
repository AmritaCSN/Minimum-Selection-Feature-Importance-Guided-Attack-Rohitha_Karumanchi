import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import csv
from src import figa_MS as figa
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
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

#Decision Tree
dt_model = DecisionTreeClassifier()
scaler = MinMaxScaler(feature_range=(0, 1))
model = make_pipeline(scaler, dt_model)
dt_result=np.array([[0] for i in range(100)]) # 2d array with 100 rows and 1 column
with open("dt_data.csv","a") as f:
    data_writer = csv.writer(f)
    for i in range(0,100):
            print(f"DT Iteration : {i}")
            X_target,y_target,X_train,y_train=select_target(data)
            dt_target_model = model.fit(X_train, y_train)
            dt_attack=figa.MS_FIGA(X_train, y_train, "permutation")
            sample=dt_attack.generate(dt_target_model,X_target,y_target,e=4.0,n=25)
            #dt_result.append(sample)
            dt_result[i][0] = sample
            sample_data=dt_result.flatten()
            data_writer.writerow(sample_data)
#flatten the 2d array and calculate the average
dt_list = dt_result.flatten()
dt_min_sample = sum(dt_list) / len(dt_list)
print(f" dt result {dt_min_sample}  and dt samples {dt_result}")
# open CSV file in append mode
with open('decision_tree_attack.csv', mode='a', newline='') as file:
    # create CSV writer object
    writer = csv.writer(file)
    # write each row of data to CSV file
    for row in dt_result:
        writer.writerow(row)


