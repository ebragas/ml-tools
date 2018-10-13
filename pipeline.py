import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

'''
Define the functions and Pipeline Transformers that will encode the data preprocessing steps
required to train and predict using our preditive model. This enables repeatability of the
process and protects against sample leakage.
'''

def one_hot(input_df, columns):
    '''
    One-hot encode the provided list of columns and return a new copy of the data frame
    '''
    df = input_df.copy()

    for col in columns:
        dummies = pd.get_dummies(df[col].str.lower())
        dummies.drop(dummies.columns[-1], axis=1, inplace=True)
        df = df.drop(col, axis=1).merge(dummies, left_index=True, right_index=True)
    
    return df


def prepare_target(input_df):
    '''
    Create features required to derive the target classification, and return a copy of the
    data frame.

    TODO: Refactor into a target transformation pipeline
    TODO: Move surge_pct feature derivation into separate pipeline
    '''
    df = input_df.copy()
    
    # Cast datetimes
    df['last_trip_dt'] = pd.to_datetime(df['last_trip_date'])
    df['signup_dt'] = pd.to_datetime(df['signup_date'])
    
    # Find max trip date
    df['today_dt'] = df['last_trip_dt'].max()
    
    # Find inactive days
    df['inactive_timedelta'] = df['today_dt'] - df['last_trip_dt']
    df['inactive_days'] = df['inactive_timedelta'].dt.days

    # Create target classes
    df.loc[df['inactive_days'] > 30, 'churned'] = 1
    df.loc[df['inactive_days'] <= 30, 'churned'] = 0
    
    # Creating a new always_surge_users column because we have 1145 customers who uses the service at surge only
    df.loc[df['surge_pct'] >= 100.0, 'always_surge_users'] = 1
    df.loc[df['surge_pct'] < 100.0, 'always_surge_users'] = 0
    
    return df


class FillTransformer(BaseEstimator, TransformerMixin):
    '''
    Impute NaN values

    # TODO: Parameterize so values can be imputed with -1, mean, median, or mode.
    '''
    def fit(self, X, y):
        self.fill_value = -1
        return self
    
    def transform(self, X):
        # paramaterize this with mean, median, mode, etc. 
        # fill with -1
        X[['avg_rating_of_driver', 'avg_rating_by_driver']].fillna(-1, axis=1, inplace=True)
        
        X.loc[:, 'luxury_car_user'] = X['luxury_car_user'].map({False: 0, True: 1})
        return X


class OneHotTransformer(BaseEstimator, TransformerMixin):
    '''
    One-hot encode features
    '''

    def fit(self, X, y):
        df = one_hot(X, ['city', 'phone'])
        self.train_columns = df.columns
        
        return self
    
    def transform(self, X):
        df = X.copy()
        df = one_hot(df, ['city', 'phone'])

        # Remove untrained columns
        for col in self.train_columns:
            if col not in df.columns:
                df[col] = 0
                
        # Add trained on columns
        for col in df.columns:
            if col not in self.train_columns:
                df.drop(col, axis=1, inplace=True)
        
        return df[self.train_columns]