import pandas as pd
import numpy as np
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import LabelEncoder

class Preprocessing():
    def __init__(self):
        self.data_root = '../html2021final/'
        self.status = pd.read_csv(self.data_root + 'status.csv')
        self.demographics = pd.read_csv(self.data_root + 'demographics.csv')
        self.location = pd.read_csv(self.data_root + 'location.csv')
        self.population = pd.read_csv(self.data_root + 'population.csv')
        self.services = pd.read_csv(self.data_root + 'services.csv')
        self.satisfaction = pd.read_csv(self.data_root + 'satisfaction.csv')

    def Status(self):
        self.status["Churn Category"] = self.status["Churn Category"].map(
            {"No Churn": 0, 
             "Competitor": 1, 
             "Dissatisfaction": 2,
             "Attitude": 3,
             "Price": 4,
             "Other": 5})

    def Demographics(self):
        data_size = self.demographics.shape[0]

        self.demographics["Gender"] = self.demographics["Gender"].map({"Male": 0, "Female": 1})
        self.demographics["Under 30"] = self.demographics["Under 30"].map({"No": 0, "Yes": 1})
        self.demographics["Senior Citizen"] = self.demographics["Senior Citizen"].map({"No": 0, "Yes": 1})
        self.demographics["Married"] = self.demographics["Married"].map({"No": 0, "Yes": 1})
        self.demographics["Dependents"] = self.demographics["Dependents"].map({"No": 0, "Yes": 1})

        self.demographics = self.demographics.drop(['Count'], axis=1)

    def Location(self):
        self.location = self.location.merge(self.population, how='left', on='Zip Code')
        self.location["Population"] = self.location["Population"].map(lambda i: np.log(abs(i) + 1))
        self.location = self.location.drop(['ID'], axis=1)

        self.location['City'] = self.location['City'].fillna('None')
        self.location['City'] = LabelEncoder().fit_transform(self.location['City'])

        data_size = self.location.shape[0]
        for i in range(data_size):
            if not np.isnan(self.location.loc[i, "Zip Code"]):
                self.location.loc[i, "Zip Code"] = self.location.loc[i, "Zip Code"] - 90000

        self.location = self.location.drop(
            ['Count', 'Country', 'State', 'Lat Long', 'Latitude', 'Longitude'], axis=1)

    def Services(self):
        data_size = self.services.shape[0]
        self.services = self.services.drop(['Count', 'Quarter'], axis=1)

        object_features = []
        for dtype, feature in zip(self.services.dtypes, self.services.columns):
            if dtype == 'object':
                object_features.append(feature)
        NotYN = ['Customer ID', 'Offer', 'Internet Type', 'Contract', 'Payment Method']
        for i in NotYN:
            object_features.remove(i)
        
        for c in object_features:
            self.services[c] = self.services[c].map({"No": 0, "Yes": 1})

        self.services['Offer'] = self.services['Offer'].map(
            {"None": 0, "Offer A": 1, "Offer B": 2, "Offer C": 3, "Offer D": 4, "Offer E": 5})
        self.services['Internet Type'] = self.services['Internet Type'].map(
            {"None": 0, "DSL": 1, "Fiber Optic": 2, "Cable": 3})
        self.services['Contract'] = self.services['Contract'].map(
            {"Month-to-Month": 0, "One Year": 1, "Two Year": 2})
        self.services['Payment Method'] = self.services['Payment Method'].map(
            {"Bank Withdrawal": 0, "Credit Card": 1, "Mailed Check": 2})

        log_col = ["Total Charges", "Total Long Distance Charges", "Total Refunds", "Total Extra Data Charges", "Total Revenue"]
        for col in log_col:
            self.services[col] = self.services[col].map(lambda i: np.log(abs(i) + 1))
        
    def PostProcessing(self):
        data_size = self.data.shape[0]
        Avg_Charges = np.zeros((data_size, 1))
        Avg_Long_Distance_Charges = np.zeros((data_size, 1))
        Avg_Refunds = np.zeros((data_size, 1))
        Avg_Extra_Data_Charges = np.zeros((data_size, 1))
        Avg_Revenue = np.zeros((data_size, 1))
        for i in range(data_size):
            Avg_Charges[i, 0] = np.exp(self.data.loc[i, "Total Charges"]) / self.data.loc[i, "Tenure in Months"]
            Avg_Long_Distance_Charges[i, 0] = np.exp(self.data.loc[i, "Total Long Distance Charges"]) / self.data.loc[i, "Tenure in Months"]
            Avg_Refunds[i, 0] = np.exp(self.data.loc[i, "Total Refunds"]) / self.data.loc[i, "Tenure in Months"]
            Avg_Extra_Data_Charges[i, 0] = np.exp(self.data.loc[i, "Total Extra Data Charges"]) / self.data.loc[i, "Tenure in Months"]
            Avg_Revenue[i, 0] = np.exp(self.data.loc[i, "Total Revenue"]) / self.data.loc[i, "Tenure in Months"]

        self.data['Avg_Charges'] = Avg_Charges
        self.data['Avg_Long_Distance_Charges'] = Avg_Long_Distance_Charges
        self.data['Avg_Refunds'] = Avg_Refunds
        self.data['Avg_Extra_Data_Charges'] = Avg_Extra_Data_Charges
        self.data['Avg_Revenue'] = Avg_Revenue

        self.data = self.data.drop(
            ["Total Charges", "Total Long Distance Charges", "Total Refunds", 
             "Total Extra Data Charges", "Total Revenue"], axis=1)
        
    def Create_files(self):
        self.Status()
        self.Demographics()
        self.Location()
        self.Services()

        training_data = pd.read_csv(self.data_root + 'Train_IDs.csv')
        testing_data = pd.read_csv(self.data_root + 'Test_IDs.csv')
        self.data = pd.concat([training_data, testing_data], axis=0, ignore_index=True)

        self.data = self.data.merge(self.demographics, how='left', on='Customer ID')
        self.data = self.data.merge(self.location, how='left', on='Customer ID')
        self.data = self.data.merge(self.services, how='left', on='Customer ID')
        self.data = self.data.merge(self.satisfaction, how='left', on='Customer ID')

        self.data_train = training_data.merge(self.data, how='left', on='Customer ID')
        self.data_train = self.data_train.drop(['Customer ID'], axis=1)
        data_train_np = self.data_train.to_numpy()
        
        self.ID = self.data['Customer ID']
        self.data = self.data.drop(['Customer ID'], axis=1)
        columns = self.data.columns
        data_np = self.data.to_numpy()

        estimator = BayesianRidge(n_iter=300, 
                                  tol=0.001, 
                                  alpha_1=1e-6, 
                                  alpha_2=1e-6, 
                                  lambda_1=1e-6, 
                                  lambda_2=1e-6,
                                  lambda_init=1e-3)
        imputer = IterativeImputer(estimator=estimator, 
                                   missing_values=np.nan, 
                                   max_iter=10, 
                                   tol=0.001, 
                                   n_nearest_features=None, 
                                   initial_strategy='mean', 
                                   imputation_order='ascending', 
                                   random_state=1126)
        imputer.fit(data_train_np)
        data_np = imputer.transform(data_np)

        self.data = pd.DataFrame(data_np, columns=columns)
        self.data = pd.concat([self.ID, self.data], axis=1)

        self.PostProcessing()

        self.data_train = self.status.merge(self.data, how='left', on='Customer ID')
        self.data_test = testing_data.merge(self.data, how='left', on='Customer ID')

        if not os.path.exists('./data'):
            os.makedirs('./data')

        self.data_train.to_csv('./data/train.csv', index=False)
        self.data_test.to_csv('./data/test.csv', index=False)


