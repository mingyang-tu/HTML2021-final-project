import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from Preprocessing import Preprocessing

class XGB_method():
    def __init__(self, 
                 random_state = 1, 
                 n_estimators = 1000, 
                 lr = 0.3, 
                 max_depth = 3, 
                 reg_lambda = 10, 
                 reg_alpha = 0.3, 
                 gamma = 0.1, 
                 min_child_weight = 1,
                 subsample = 1,
                 colsample_bytree = 1,
                 preprocess=False):
        if preprocess:
            p = Preprocessing()
            p.Create_files()
        self.data_train = pd.read_csv('./data/train.csv')
        self.data_test = pd.read_csv('./data/test.csv')

        self.df_X, self.df_Y, self.df_test, self.test_id = self.TrainTestSplit(self.data_train, self.data_test)

        self.random_state = random_state
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

    def TrainTestSplit(self, train, test):
        train_Y = train['Churn Category'].to_numpy()
        train_X = train.drop(['Customer ID', 'Churn Category'], axis=1).to_numpy()
        
        test_id = test.loc[:, 'Customer ID']
        test_X = test.drop(['Customer ID'], axis=1).to_numpy()

        return train_X, train_Y, test_X, test_id

    def f1score(self, prediction, target, label):
        size = len(prediction)
        TP = 0
        FP = 0
        FN = 0
        for i in range(size):
            if (target[i] == label) and (prediction[i] == label):
                TP += 1
            elif (target[i] != label) and (prediction[i] == label):
                FP += 1
            elif (target[i] == label) and (prediction[i] != label):
                FN += 1
        if (2 * TP + FP + FN > 0):
            return (2 * TP) / (2 * TP + FP + FN)
        else: 
            return 0
        
    def Calculate_Average_F1score(self, prediction, target, show=False):
        f1 = np.zeros(6)
        for i in range(6):
            f1[i] = self.f1score(prediction, target, i)
        d = {'F1-scores': f1}
        stat = pd.DataFrame(data=d)
        avg = np.mean(f1)
        
        if show:
            print("===== F1-scores =====")
            print(stat)
            print(f"Avg = {avg:.3f}")
            print("=====================")
            
        return avg

    def Five_Cross_Validation(self):
        random_state = self.random_state
        n_estimators = self.n_estimators
        lr = self.lr
        max_depth = self.max_depth
        reg_lambda = self.reg_lambda
        reg_alpha = self.reg_alpha
        gamma = self.gamma
        min_child_weight = self.min_child_weight
        subsample = self.subsample
        colsample_bytree = self.colsample_bytree
        print("\n===== Hyperparameters =====")
        print(f"random_state = {random_state}")
        print(f"n_estimators = {n_estimators}")
        print(f"lr = {lr}")
        print(f"max_depth = {max_depth}")
        print(f"reg_lambda = {reg_lambda}")
        print(f"reg_alpha = {reg_alpha}")
        print(f"gamma = {gamma}")
        print(f"min_child_weight = {min_child_weight}")
        print(f"subsample = {subsample}")
        print(f"colsample_bytree = {colsample_bytree}")
        print("==========================\n")

        data_size = self.df_X.shape[0]
        f1_score_train = 0
        f1_score_val = 0
        num_of_iter = 5
            
        for fold in range(num_of_iter):
            print(f"===== Fold: {fold} =====")
            X_train = self.df_X[np.array([i for i in range(data_size) if i % 5 != fold]), :]
            y_train = self.df_Y[np.array([i for i in range(data_size) if i % 5 != fold])]
            X_val = self.df_X[np.array([i for i in range(data_size) if i % 5 == fold]), :]
            y_val = self.df_Y[np.array([i for i in range(data_size) if i % 5 == fold])]

            model = XGBClassifier(random_state=random_state, 
                                  n_estimators=n_estimators, 
                                  learning_rate=lr, 
                                  max_depth=max_depth, 
                                  booster='gbtree', 
                                  reg_lambda=reg_lambda,
                                  reg_alpha=reg_alpha, 
                                  gamma=gamma, 
                                  min_child_weight=min_child_weight,
                                  subsample=subsample,
                                  colsample_bytree=colsample_bytree,
                                  eval_metric='mlogloss', 
                                  use_label_encoder=False)

            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)

            f1_score_train += self.Calculate_Average_F1score(y_pred_train, y_train, show=True)
            f1_score_val += self.Calculate_Average_F1score(y_pred_val, y_val, show=True)
            print()

        f1_score_train = f1_score_train / 5
        f1_score_val = f1_score_val / 5

        print(f"Training f1-score = {f1_score_train:.4f}")
        print(f"Validation f1-score = {f1_score_val:.4f}")

    def Prediction(self):
        random_state = self.random_state
        n_estimators = self.n_estimators
        lr = self.lr
        max_depth = self.max_depth
        reg_lambda = self.reg_lambda
        reg_alpha = self.reg_alpha
        gamma = self.gamma
        min_child_weight = self.min_child_weight
        subsample = self.subsample
        colsample_bytree = self.colsample_bytree
        print("\n===== Hyperparameters =====")
        print(f"random_state = {random_state}")
        print(f"n_estimators = {n_estimators}")
        print(f"lr = {lr}")
        print(f"max_depth = {max_depth}")
        print(f"reg_lambda = {reg_lambda}")
        print(f"reg_alpha = {reg_alpha}")
        print(f"gamma = {gamma}")
        print(f"min_child_weight = {min_child_weight}")
        print(f"subsample = {subsample}")
        print(f"colsample_bytree = {colsample_bytree}")
        print("==========================\n")

        model = XGBClassifier(random_state=random_state, 
                              n_estimators=n_estimators, 
                              learning_rate=lr, 
                              max_depth=max_depth, 
                              booster='gbtree', 
                              reg_lambda=reg_lambda,
                              reg_alpha=reg_alpha, 
                              gamma=gamma, 
                              min_child_weight=min_child_weight,
                              subsample=subsample, 
                              colsample_bytree=colsample_bytree,
                              eval_metric='mlogloss', 
                              use_label_encoder=False)
        model.fit(self.df_X, self.df_Y)
        y_pred = model.predict(self.df_test)

        sub = pd.DataFrame({'Customer ID': self.test_id, 'Churn Category': y_pred})
        sub.to_csv('Prediction.csv', index=False)
        print("Save Prediction File as [Prediction.csv]")


