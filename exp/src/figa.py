import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.metrics import recall_score
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline


class FeatureImportanceGuidedAttack:
    def __init__(self, X_train, y_train, feat_imp_method):
        """
        TODO: write a docstring for this class
        """

        valid_feat_importances = [
            "gini",
            "info_gain",
            "rfe",
            "permutation",
            "sfs",
            "random",
        ]

        assert (isinstance(X_train, pd.DataFrame)) and (
            isinstance(y_train, pd.Series)
        ), "X_train and y_train should be a pandas dataframe and pandas Series respectively"
        assert (
            X_train.shape[0] == y_train.shape[0]
        ), "X_train and y_train need to have the same number of rows"
        assert feat_imp_method in valid_feat_importances, (
            f"Given feature importance method: {feat_imp_method} is not supported"
            f"Please select from: {valid_feat_importances}"
        )

        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.feat_imp_method = feat_imp_method
        self.scaler = MinMaxScaler()
        self.fit()

    def fit(self):
        """
        Creates a dataframe of features in sorted order of their decreasing importance
        and their corresponding importances and attack direction
        """
        X_scaled = self.scaler.fit_transform(self.X_train)

        if self.feat_imp_method == "info_gain":
            mutual_info = mutual_info_classif(X_scaled, self.y_train)
            feat_importance = pd.Series(mutual_info)
            sorted_feature = pd.DataFrame(
                {"feature": self.X_train.columns, "importance": feat_importance}
            )

        elif self.feat_imp_method == "gini":
            feature_scores = (
                DecisionTreeClassifier()
                .fit(X_scaled, self.y_train)
                .feature_importances_
            )
            sorted_feature = pd.DataFrame(
                {"feature": self.X_train.columns, "importance": feature_scores}
            )

        elif self.feat_imp_method == "rfe":
            rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=20)
            rfe.fit(X_scaled, self.y_train)
            sorted_feature = pd.DataFrame(
                {"feature": self.X_train.columns, "importance": rfe.ranking_}
            )

        elif self.feat_imp_method == "permutation":
            X_train_new, X_val, y_train_new, y_val = train_test_split(
                self.X_train, self.y_train, test_size=0.15, random_state=42
            )
            clf = DecisionTreeClassifier().fit(X_train_new, y_train_new)
            result = permutation_importance(
                clf, X_val, y_val, n_repeats=30, random_state=42, n_jobs=-1
            )
            sorted_feature = pd.DataFrame(
                {"feature": self.X_train.columns, "importance": result.importances_mean}
            )

        elif self.feat_imp_method == "sfs":
            sfs = SequentialFeatureSelector(
                estimator=DecisionTreeClassifier(),
                k_features=10,
                forward=True,
                scoring="f1",
                n_jobs=-1,
            )
            sfs.fit(X_scaled, self.y_train)
            sorted_feature = pd.DataFrame(
                {"feature": self.X_train.columns, "importance": sfs.k_score_}
            )

        elif self.feat_imp_method[0:6] == "random":
            random_order = np.arange(1, len(self.X_train.columns) + 1)
            np.random.shuffle(random_order)
            sorted_feature = pd.DataFrame(
                {"feature": self.X_train.columns, "importance": random_order}
            )

        X_scaled = pd.DataFrame(
            data=X_scaled, index=self.X_train.index, columns=self.X_train.columns
        )

        phish_mean = X_scaled.loc[self.y_train == 1].mean(axis=0)  ## class_to_perturb
        leg_mean = X_scaled.loc[self.y_train == 0].mean(axis=0)  ## target class

        attack_direction = np.sign(leg_mean - phish_mean)  # returns a pd.Series

        sorted_feature["direction"] = attack_direction.values

        if (
            self.feat_imp_method == "gini"
            or self.feat_imp_method == "info_gain"
            or self.feat_imp_method == "permutation"
            or self.feat_imp_method == "sfs"
        ):
            sorted_feature.sort_values(by=["importance"], ascending=False, inplace=True)

        elif self.feat_imp_method == "rfe" or self.feat_imp_method[0:6] == "random":
            sorted_feature.sort_values(by=["importance"], ascending=True, inplace=True)

        # pandas dataframe with 3 columns, feature, importance, direction
        # sorted attacked direction
        self.sorted_attack_dir = sorted_feature
        

    def generate(self, X, y, e, n):
        """
        Creates adversarial samples from the dataset samples.
        Args:
            X (dataframe): Dataset to attack (every sample will be perturbed)
            y (dataframe): Y or label dataframe sample used by the attack
            e (float): Value between 0 and 1. Specifies how much a sample should be modified
            n (integer): Specifies the number of features to be modified
        Returns:
            dataframe : Contains the adversarial modified samples and legitimate samples which corresponds to the y label dataframe.
        """

        assert (isinstance(X, pd.DataFrame)) and (
            isinstance(y, pd.Series)
        ), "X and y should be a pandas Dataframe  and pandas Series respectively"
        assert X.shape[0] == y.shape[0], "X and y need to have the same number of rows"

        # creating a scaled data
        X_scaled = self.scaler.transform(X)
        # convert the X_scaled to dataframe
        X_scaled = pd.DataFrame(data=X_scaled, index=X.index, columns=X.columns)

        # list of features to be modified and their feature direction , datatype , min and max values
        feature_mod = self.sorted_attack_dir["feature"][:n].to_list()
        feature_direction_value = self.sorted_attack_dir["direction"][:n].to_list()

        # epsilon budget
        real_e = X_scaled.loc[y == 1].sum(axis=1) * e
        e_budget = real_e / n
        phish_index_values = e_budget.index  # hack to get the target indexes

        # type cast variables into numpy array and reshaping it for matrix multiplication
        e_budget_values = e_budget.values.reshape(-1, 1)
        feature_direction_value = np.array(feature_direction_value).reshape(1, -1)

        # Matrix Multiplication and adding the perturbations
        self.perturbation_matrix = e_budget_values @ feature_direction_value
        X_scaled.loc[phish_index_values, feature_mod] += self.perturbation_matrix

        # clipping features to observe min/max constraints of the original dataset
        # this will help ensure data feasibility
        for feature in feature_mod:
            X_scaled.loc[X_scaled[feature] < 0, feature] = 0
            X_scaled.loc[X_scaled[feature] > 1, feature] = 1

        # representing the data back in the original scale
        X_attack = self.scaler.inverse_transform(X_scaled)

        X_attack = pd.DataFrame(data=X_attack, index=X.index, columns=X.columns)

        # Maintain data consistency - if it was discrete before, it should be discrete after
        # if datatype is int and direction is positive round down else if direction is negative round up.
        feature_direction_value = feature_direction_value.reshape(n)

        for i in range(n):
            # look in the first row and check the datatype of that feature
            ## TODO should be way to check the datatype of the whole columns
            if isinstance(
                self.X_train.loc[self.X_train.index[0], feature_mod[i]],
                (np.int, np.int32, np.int64),
            ):
                if feature_direction_value[i] == 1:
                    X_attack.loc[phish_index_values, feature_mod[i]] = np.floor(
                        X_attack.loc[phish_index_values, feature_mod[i]]
                    )
                else:
                    X_attack.loc[phish_index_values, feature_mod[i]] = np.ceil(
                        X_attack.loc[phish_index_values, feature_mod[i]]
                    )
        return X_attack