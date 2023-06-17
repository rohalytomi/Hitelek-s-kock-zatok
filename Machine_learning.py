import pandas as pd
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
df_etf_returns=get_etf_returns('BND','log','Adj Close')
#def calculate_squared_returns(self):
    #self_data['squared_returns'] = self_data['returns'] ** 2
df_squared_returns=df_etf_returns**2

df_etf_returns.rename(columns={'BND': 'returns'}, inplace=True)
df_squared_returns=df_etf_returns.dropna(inplace=True)
print(df_etf_returns)