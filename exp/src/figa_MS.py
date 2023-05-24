import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.cluster import KMeans
from src import figa as figa
    

class MS_FIGA(figa.FeatureImportanceGuidedAttack):
    def __init__(self, X_train, y_train, feat_imp_method):
        super().__init__(X_train, y_train, feat_imp_method)
        self.centroids=None
        self.closest_samples=None
        self.closest_sample_indexes=None
        self.feature_mod=None
        self.legit_top_features=None
        self.sample=None
        self.data=pd.concat([self.X_train, self.y_train], axis=1)
        self.legit_data=self.data[self.data['label'] == 0]
        self.legit_data=self.legit_data.drop("label", axis=1)
        self.phish_data=self.data[self.data['label'] == 1]
        self.phish_data=self.phish_data.drop("label", axis=1)
        
        
    
    def kmeans(self,clusters,cluster_data):
        # Initialize the KMeans algorithm
        kmeans = KMeans(n_clusters=clusters)
        # Fit the KMeans algorithm to the features
        kmeans.fit(cluster_data)
        #getting centroids
        self.centroids = kmeans.cluster_centers_

        # Compute distances between samples and centroids
        distances = np.linalg.norm(cluster_data[:, np.newaxis, :] - self.centroids, axis=2)

        # Find the index of closest sample for each centroid
        self.closest_sample_indexes = np.argsort(distances, axis=0)[:10]

        # Get 1D array of 2 indexes of samples for each centroid
        top_2_indexes = self.closest_sample_indexes.flatten()[:2 * self.closest_sample_indexes.shape[0]]
        self.closest_samples= pd.DataFrame(self.legit_data.iloc[top_2_indexes])



    def attack(self,X, y, target_model, X_target, y_target, e, n):
            
            attack = figa.FeatureImportanceGuidedAttack(X, y, self.feat_imp_method)

            #feature importance
            self.feature_mod=attack.sorted_attack_dir["feature"].iloc[:n].to_list()

            #clustering
            legit_top_features=self.legit_data[self.feature_mod]
            legit_top_features=self.scaler.fit_transform(legit_top_features)
            self.kmeans(n,legit_top_features)

            X_attack = attack.generate(X_target, y_target, e, n)
            pred_y = target_model.predict(X_attack)
            recall = recall_score(y_target, pred_y)
            #print(f'recall is {recall} for sample {self.sample}')
                
            return recall

    def isSuccess(self,recall,target_model, X_target,y_target,e, n):
            while recall ==1 and self.sample<10000:
                #print(f"sample {self.sample} and n = {n}, comparision{self.sample < n}")
                #print(self.X_train_attack)
                #print(self.y_train_attack)
                
                if self.feat_imp_method=="info_gain" and self.sample<3:
                     self.X_train_attack = self.X_train_attack.append(self.closest_samples.iloc[self.sample])
                     self.legit_data= self.legit_data.drop(self.X_train_attack.index[-1])

                     self.y_train_attack.at[self.X_train_attack.index[-1]] = 0
                
                elif self.feat_imp_method=="info_gain" and self.sample==3:
                     self.X_train_attack = self.X_train_attack.append(self.closest_samples.iloc[self.sample])
                     self.legit_data= self.legit_data.drop(self.X_train_attack.index[-1])

                     self.y_train_attack.at[self.X_train_attack.index[-1]] = 0
                     recall = self.attack(self.X_train_attack, self.y_train_attack,target_model, X_target,y_target, e, n)
               
                     
                else:
                    self.X_train_attack = self.X_train_attack.append(self.closest_samples.iloc[0])
                    self.legit_data= self.legit_data.drop(self.X_train_attack.index[-1])

                    self.y_train_attack.at[self.X_train_attack.index[-1]] = 0
                    recall = self.attack(self.X_train_attack, self.y_train_attack,target_model, X_target,y_target, e, n)
                
                
                self.sample=self.sample+1
            
            return self.sample


    def generate(self, target_model,X_target, y_target, e, n):

        legit_data_transformed= self.scaler.fit_transform(self.legit_data)
        self.kmeans(n,legit_data_transformed)

        self.X_train_attack = pd.DataFrame(None,columns=self.feature_mod) 
        self.X_train_attack = self.X_train_attack.append(self.phish_data.sample(1))
        self.y_train_attack=  pd.Series(1, index=self.X_train_attack.index)
        #print(self.X_train_attack)
        #print(self.y_train_attack)
       
        self.sample = 0
        recall = 1
        self.sample=self.isSuccess(recall,target_model,X_target,y_target, e, n)
        return self.sample

    

        





