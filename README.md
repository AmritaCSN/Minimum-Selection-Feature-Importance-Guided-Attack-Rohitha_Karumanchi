# Minimum-Selection-Feature-Importance-Guided-Attack

Feature Importance Guided Attack is a novel adversarial evasion attack against tabular datasets. It works by ranking the features in order of their importance and then perturbing the most important features in the direction of the target class.

Ref: https://arxiv.org/pdf/2106.14815.pdf

FIGA assumes the knowledge of the total available data for training. In a typical case, the attacker will query a target model (an oracle) to label their training data and perform an adversarial attack on the labeled data. The notion that the attacker could use the target model to label a dataset is infeasible as it requires an excessive amount of queries to the target model. The project work MS_FIGA is a novel methodology to train FIGA with reduced training samples. This method leverages an intuitive approach that utilizes the KMeans clustering algorithm to select legitimate samples from the training data for querying. 
The attack MS_FIGA reduces the data required to train the FIGA attack model by selecting the samples using KMeans clustering. It differs from the original FIGA algorithm in three ways, it perturbs each phishing sample individually, and iteratively perturb the sample until it successfully evades the target model, we also minimize the number of legitimate samples and use clustering to select an evenly distributed set of samples.

**figa.py**: Holds the original FIGA attack code. 
Input: Training data(X_train, y_train), feature importance method(feat_imp_method), Data that needs to be perturbed(X,y), perturbation(e), number of features(n).
Output: Perturbed data(X_attack)
Working: Calculates feature importance rankings and attack direction. Generates attack samples.

**MS_figa.py**: Triggers FIGA attack iteratively with least number of samples possible.
Input: Training data(X,y), feature importance method(feat_imp_method), Data that needs to be perturbed(X_target, y_target), Target model(target_model),  perturbation(e), number of features(n)
Output: Number of Samples that successfully triggered the attack.
Working: Seperates legitimate and malicious samples from the total dataset. Triggers FIGA attack iteratively by using kmeans clustering.

decision_tree_attack, logistic_regression_attack, mlp_attack, random_forest_attack, and xg_boost_attack files contain the code that triggers the attack on the respective trained models.

**FIGA_data**: is the dataset used.
Number of samples: 348739
Number of features: 52
Phishing samples: 138473
Legitimate samples: 210266

