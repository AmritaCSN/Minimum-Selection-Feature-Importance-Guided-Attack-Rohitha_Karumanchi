from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")
from src import figa_MS as figa
import pandas as pd
import numpy as np
import csv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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


#MLP
MLP_model = MLPClassifier(solver='adam',hidden_layer_sizes=(50,),max_iter=300,random_state=42)
#MLP_model.fit(X_train,y_train)
scaler = StandardScaler()
model = make_pipeline(scaler, MLP_model)
mlp_result=np.array([[0] for i in range(2)]) # 2d array with 100 rows and 1 column
with open("mlp_data.csv","a") as f:
    data_writer = csv.writer(f)
    for i in range(0,2):
        X_target,y_target,X_train,y_train=select_target(data)
        MLP_target_model = model.fit(X_train, y_train)
        MLP_attack=figa.MS_FIGA(X_train, y_train, "info_gain")
        sample=MLP_attack.generate(MLP_target_model,X_target,y_target,e=0.57,n=7)
        #mlp_result.append(sample)
        mlp_result[i][0] = sample
        sample_data=mlp_result.flatten()
        data_writer.writerow(sample_data)
#flatten the 2d array and calculate the average
mlp_list = mlp_result.flatten()
mlp_min_sample = sum(mlp_list ) / len(mlp_list )
print(f" mlp result {mlp_min_sample}  and mlp samples {mlp_result}")
# open CSV file in append mode
with open('mlp_attack.csv', mode='a', newline='') as file:
    # create CSV writer object
    writer = csv.writer(file)
    # write each row of data to CSV file
    for row in mlp_result:
        writer.writerow(row)
            