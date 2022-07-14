import pandas as pd
pd.set_option('mode.chained_assignment', None)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from utils import replace_yes_no_with_boolean
import numpy as np


class ChurnClassifier:

    def __init__(self):
        self.df = pd.read_csv("data.csv") #We read the dataset with information about customers
        self.df = self.df.rename(columns = {'tenure':'MonthInCompany'}) #For readibility, we rename this column

        del self.df['customerID'] #Unnecessary variable

    def __imputing_median_over_whitespaces(self, data):
        try:
            return float(data)
        except:
            return np.median(self.df[self.df['TotalCharges'] != ' ']['TotalCharges'].astype(float))

    def __string_to_boolean(self):
        self.df['Partner'] = self.df['Partner'].apply(replace_yes_no_with_boolean)
        self.df['gender'] = self.df['gender'].apply(replace_yes_no_with_boolean)
        self.df['Dependents'] = self.df['Dependents'].apply(replace_yes_no_with_boolean)
        self.df['PhoneService'] = self.df['PhoneService'].apply(replace_yes_no_with_boolean)
        self.df['PaperlessBilling'] = self.df['PaperlessBilling'].apply(replace_yes_no_with_boolean)
        self.df['Churn'] = self.df['Churn'].apply(replace_yes_no_with_boolean)

    def __generating_examples_minority_class(self):
        #We create random samples for the minority class with K nearest neighbor technique like SMOTE
        sm = SMOTE()
        x_resampled, y_resampled = sm.fit_resample(self.X,self.y)
        df_resampled = x_resampled.join(y_resampled)
        
        return df_resampled

    def preproccesing(self):
        #Changing boolean answers for 1 and 0
        self.__string_to_boolean()

        #For cases which we don´t have the value of this variable, we replace it with the median of the variable
        self.df['TotalCharges'] = self.df['TotalCharges'].apply(self.__imputing_median_over_whitespaces)

        #We define the target variable
        self.y = self.df[['Churn']] 

        #We drop the target variable from our dataset because now it will be our training dataset
        del self.df['Churn']

        #This are the quantitative columns
        quantitative_features = self.df.select_dtypes(['int','float'])

        #This are the qualitative columns
        qualitative_features = self.df.select_dtypes(['object'])

        #We scale the quantitative variable which have big numbers
        scaler = StandardScaler()
        quantitative_features[['MonthlyCharges','TotalCharges']] = scaler.fit_transform(quantitative_features[['MonthlyCharges','TotalCharges']])
        self.df = pd.DataFrame(quantitative_features, columns = self.df.select_dtypes(['int','float']).columns)

        #We do One Hot Encoder with out categorical variables and join them with our quantitative and scaled variables
        self.df = self.df.join(pd.get_dummies(qualitative_features).reset_index(drop=True))

        #Now we have our features, before we separed the y variable
        self.X = self.df

        #Our dataset is unbalanced so we should to make some adjusts in order to have the same amount of examples for the two classes
        self.df_resampled = self.__generating_examples_minority_class()

    def training(self):
        #We will use the Random Forest Classifier
        self.rf = RandomForestClassifier()

        #Split the dataset into training and testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df_resampled.drop(columns = 'Churn'),self.df_resampled[['Churn']], test_size = 0.2, random_state= 21)

        #Fitting the model
        self.rf.fit(self.X_train, self.y_train.values.ravel()) #We put values because we want the values of the column and ravel convert it in a flatten array

    def testing(self):
        print(f"The accuracy of the model is around {round(self.rf.score(self.X_test, self.y_test), 2)*100} %")

    def predict(self, examples):
        return self.rf.predict(examples)
