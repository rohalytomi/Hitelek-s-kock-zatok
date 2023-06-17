import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np
import os
bnd = 'BND'
dba = 'DBA'
weights = {'dba':0.5, 'bnd':0.5}
conf = [0.95]
def read_etf_file(etf):
    filename = os.path.join(etf + '.csv')
    df = pd.read_csv(filename, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df

def get_etf_returns(etf_name,
    return_type='log', fieldname='Adj Close'):
    df = read_etf_file(etf_name)
    df = df[[fieldname]]
    df['shifted'] = df.shift(1)
    if return_type=='log':
        df['return'] = np.log(df[fieldname]/df['shifted'])
    if return_type=='simple':
        df['return'] = df[fieldname]/df['shifted']-1
    # restrict df to result col
    df = df[['return']]
    # rename column
    df.columns = [etf_name]
    # df = df.rename(by=col, {'return': etf_name})
    return df
df_etf=get_etf_returns('BND','log','Adj Close')
df_etf=df_etf.dropna(inplace=True)
df_etf=pd.DataFrame(df_etf, columns=['returns'])
df_etf.rename(columns={'BND': 'returns'}, inplace=True)

class VariancePredictionModel:
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size

    def calculate_squared_returns(self):
        self.data['squared_returns'] = self.data['returns'] ** 2

    def create_lagged_returns(self):
        for i in range(1, self.window_size + 1):
            self.data[f'lag_{i}'] = self.data['returns'].shift(i)

    def predict_variance(self):
        X = pd.DataFrame()
        for lag in range(1, self.window_size + 1):
            X[f'Lagged_Returns_{lag}'] = self.data[f'lag_{lag}']
        X = X[self.window_size:]
        y = self.data['squared_returns'][self.window_size:]

        tscv = TimeSeriesSplit(n_splits=5)
        mse_scores = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse_scores.append(mean_squared_error(y_test, y_pred))

        average_mse = sum(mse_scores) / len(mse_scores)
        return average_mse


# Assuming you have downloaded the ETF historical price data and stored it in a DataFrame called df_etf
model = VariancePredictionModel(df_etf, window_size=20)
model.calculate_squared_returns()
model.create_lagged_returns()
mse = model.predict_variance()

#print(f"Mean Squared Error (MSE): {mse}")
print(df_etf)