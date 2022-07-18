from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

class New_features(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, df):
        df['house_by_pop'] = df["households"]/df["population"]
        df['rooms_by_pop'] = df["total_rooms"]/df["population"]
        df['rooms_by_houses'] = df["total_rooms"]/df["households"]
        # Combining longitude and latitude
        df['ll'] = df['longitude'] + df['latitude']
        #df['is_inland'] = 1*(df['ocean_proximity']=='INLAND')
        #df['<1H OCEAN'] = 1*(df['ocean_proximity']=='<1H OCEAN')
        
        df.drop(columns=[#'longitude','latitude',
                         #'ocean_proximity',
                         'households','population','total_bedrooms'], inplace=True)
        return df

class Drop_target(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, df):
        return df[df['median_house_value'] != df['median_house_value'].max()]

class Drop_outliers(BaseEstimator, TransformerMixin):
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

class Drop_all_outliers(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        self.df = df
        Q1 = self.df.quantile(0.25)
        Q3 = self.df.quantile(0.75)
        IQR = Q3 - Q1
        self.lower_limit = Q1 - 1.5*IQR
        self.upper_limit = Q3 + 1.5*IQR
        self.index = self.lower_limit.index
        return self

    def transform(self, df):
        for i in self.index:
            df = df[(df[i]>self.lower_limit[i]) & (df[i]<self.upper_limit[i])]
        return df