import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error, mean_squared_error

DUMMY_FEATS = {'UsageBand', 'fiModelDesc',
                'fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries',
                'fiModelDescriptor', 'ProductSize', 'fiProductClassDesc',
                'state', 'ProductGroup', 'ProductGroupDesc',
                'Drive_System', 'Enclosure', 'Forks',
                'Pad_Type', 'Ride_Control', 'Stick',
                'Transmission', 'Turbocharged', 'Blade_Extension',
                'Blade_Width', 'Enclosure_Type', 'Engine_Horsepower',
                'Hydraulics', 'Pushblock', 'Ripper',
                'Scarifier', 'Tip_Control', 'Tire_Size',
                'Coupler', 'Coupler_System', 'Grouser_Tracks',
                'Hydraulics_Flow', 'Track_Type', 'Undercarriage_Pad_Width',
                'Stick_Length', 'Thumb', 'Pattern_Changer',
                'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type',
                'Travel_Controls', 'Differential_Type', 'Steering_Controls'}


def rmsle(y_true, y_pred): 
    """Compute the Root Mean Squared Log Error of the y_pred and y_true values"""
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def rmse(y_true, y_pred): 
    """Compute the Root Mean Squared Error of the y_pred and y_true values"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def one_hot(input_df, columns):
    df = input_df.copy()

    for col in columns:
        dummies = pd.get_dummies(df[col].str.lower())
        dummies.drop(dummies.columns[-1], axis=1, inplace=True)
        df = df.drop(col, axis=1).merge(dummies, left_index=True, right_index=True)
    
    return df

def fill_features(df, features):
    """
    Standardize data frame to a list of given features. Features in DataFrame but
    not in features list will be dropped. Features not in the dataframe but in the
    list will be created.
    """
    existing = df.columns

    # Drop untrained features
    for feat in existing:
        if feat not in features:
            df.drop(feat, axis=1, inplace=True)

    # Add missing features
    for feat in features:
        if feat not in existing:
            df[feat] = 0

    # Return with column selection for ordering
    return df[features]


# TODO: Feature clean-up -> input: x array -> cleaned x array
# def feature_eng(x_df):
#     x_df.get_dummies
def clean_features(dataframe, features, target=None, fill=None):
    """Input features DataFrame, clean features by columns, 
    and return cleaned DataFrame.
    
    dataframe:      pd.DataFrame containing all raw features including target.
    features:       list; features to clean.
    target:         str; name of target column.
    fill:  list; names of features to fill in zeros for if corresponding
                        columns not created during dummy variable creation.
    
    Returns: X_df; DataFrame of cleaned features
             Y_df; DataFrame of target variable
    """

    # Copy and split data frames
    X_df = dataframe[features].copy()
    if target:
        y_df = dataframe[target].copy()

    # Create dummy features
    dummies = DUMMY_FEATS.intersection(set(features))
    if dummies:
        X_df = one_hot(X_df, dummies)
    
    # Fill missing dummy features
    if fill:
        X_df = fill_features(X_df, fill)

    # Replace YearMade == 1000 with NaN
    if 'YearMade' in features:
        X_df.loc[X_df['YearMade'] == 1000, 'YearMade'] = X_df.loc[X_df['YearMade'] > 1000, 'YearMade'].median()

    # Parse year from datetime sold
    if 'saledate' in features:
        X_df['SaleYear'] = pd.to_datetime(X_df['saledate']).dt.year
        X_df['SaleMonth'] = pd.to_datetime(X_df['saledate']).dt.month
        X_df.drop('saledate', axis=1, inplace=True)

    ## All features
    # Impute NaN values with median
    X_df.fillna(X_df.median(axis=0), axis=0, inplace=True)

    if target:
        return X_df, y_df
    else:
        return X_df





if __name__ == "__main__":
    
    df = pd.read_csv('/Users/ericbragas/galvanize/dsi-regression-case-study/data/Train.csv')
    
    features = [
        'SalesID' ,
        'MachineID' ,
        'ModelID' ,
        # 'datasource' ,
        'auctioneerID' ,
        'YearMade' ,
        # 'MachineHoursCurrentMeter',
        # 'UsageBand',
        # 'saledate',    # BE CAREFUL
        'fiModelDesc',
        # 'fiBaseModel',
        # 'fiSecondaryDesc',
        # 'fiModelSeries',
        # 'fiModelDescriptor',
        # 'ProductSize',
        # 'fiProductClassDesc',
        # 'state',
        # 'ProductGroup',
        # 'ProductGroupDesc',
        # 'Drive_System',
        # 'Enclosure',
        # 'Forks',
        # 'Pad_Type',
        # 'Ride_Control',
        # 'Stick',
        # 'Transmission',
        # 'Turbocharged',
        # 'Blade_Extension',
        # 'Blade_Width',
        # 'Enclosure_Type',
        # 'Engine_Horsepower',
        # 'Hydraulics',
        # 'Pushblock',
        # 'Ripper',
        # 'Scarifier',
        # 'Tip_Control',
        # 'Tire_Size',
        # 'Coupler',
        # 'Coupler_System',
        # 'Grouser_Tracks',
        # 'Hydraulics_Flow',
        # 'Track_Type',
        # 'Undercarriage_Pad_Width',
        # 'Stick_Length',
        # 'Thumb',
        # 'Pattern_Changer',
        # 'Grouser_Type',
        # 'Backhoe_Mounting',
        # 'Blade_Type',
        # 'Travel_Controls',
        # 'Differential_Type',
        # 'Steering_Controls'
    ]

    X_df, y_df = clean_features(df, ['YearMade', 'MachineID', 'ModelID', 'MachineHoursCurrentMeter', 'UsageBand'], 'SalePrice')
