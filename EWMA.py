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

def calculate_ewma_variance(df_etf_returns, decay_factor, window):
    ewma = df_etf_returns.ewm(span=window, min_periods=window).var(decay_factor)
    return ewma

# Assuming you have downloaded the ETF historical price data and stored it in a DataFrame called df_etf
# You can calculate the EWMA variance with the given decay factors and window size:
ewma_0_94 = calculate_ewma_variance(df_etf_returns, 0.94, 100)
ewma_0_97 = calculate_ewma_variance(df_etf_returns, 0.97, 100)

# Plotting the results
import matplotlib.pyplot as plt

plt.plot(ewma_0_94, label='EWMA Variance (0.94)')
plt.plot(ewma_0_97, label='EWMA Variance (0.97)')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Variance')
plt.title('EWMA Variance of ETF Returns')
plt.show()
print(ewma_0_94)
print(ewma_0_97)