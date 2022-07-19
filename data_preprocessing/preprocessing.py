from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class New_features(BaseEstimator, TransformerMixin):
    """This function creates new features out of the old ones"""
    def __init__(self, feature_names=None):
        self.feature_names = feature_names 
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df2 = X.copy()
        df2['house_by_pop'] = X["households"]/X["population"]
        df2['rooms_by_pop'] = X["total_rooms"]/X["population"]
        df2['rooms_by_houses'] = X["total_rooms"]/X["households"]
        
        # Combining longitude and latitude
        df2['ll'] = X['longitude'] + X['latitude']
        
        #df['is_inland'] = 1*(df['ocean_proximity']=='INLAND')
        #df['<1H OCEAN'] = 1*(df['ocean_proximity']=='<1H OCEAN')
        
        df2.drop(columns=[#'longitude','latitude',
                         #'ocean_proximity',
                         'households','population','total_bedrooms'], inplace=True)
        return df2
    
class ModSwitcher(BaseEstimator):
    def __init__(self, estimator = RandomForestRegressor()):
        self.estimator = estimator


    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self


    def predict(self, X, y=None):
        return self.estimator.predict(X)


    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


    def score(self, X, y):
        return self.estimator.score(X, y)

class Drop_target(BaseEstimator, TransformerMixin):
    """To drop the rows where target value is maximal"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, df):
        return df[df['median_house_value'] != df['median_house_value'].max()]

class Drop_outliers(BaseEstimator, TransformerMixin):
    """drop rows with outliers"""
    def fit(self, df, y=None):
        self.df = df
        
        X_train, X_test = train_test_split(self.df.drop(columns='median_house_value'), test_size=.2, random_state=0)
        std7 = X_train.max() - (6.1*X_train.std() + X_train.quantile(q=0.75)) 
        self.index = std7[std7>0].index
        
        self.max_threshold = {}
        for i in self.index:
            self.max_threshold[i] = X_train[i].quantile(0.99)
        # I scaled my data beforehand
        return self

    def transform(self, df):
        for i in self.index:
            df = df[df[i]<self.max_threshold[i]]
        return df

class Trim_outliers(BaseEstimator, TransformerMixin):
    """Replace outliers by NA or by the max value"""
    def __init__(self,factor=6.1, na=False):
        self.factor = factor
        self.na = na
        
    def fit(self, X, y=None):
        
        std7 = X.max() - (self.factor*X.std() + X.quantile(q=0.75)) 
        self.index = std7[std7>0].index
        
        self.max_threshold = {}
        for i in self.index:
            self.max_threshold[i] = X[i].quantile(0.99)
        
        return self

    def transform(self, X):
        X2 = X.copy()
        if self.na:
            for i in self.index:
                X2[X2[i]>self.max_threshold[i]] = np.nan
        else:
            for i in self.index:
                X2[X2[i]>self.max_threshold[i]] = self.max_threshold[i]
        return X2