import pandas as pd
import numpy as np
import os
from scipy.stats import norm
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

def get_total_return(etf, return_type='log'):
    return get_etf_returns(etf, return_type, 'Adj Close')

def get_dividend_return(etf, return_type='log'):
    # 1 calc total simple return from Adj Close and Close
    df_ret_from_adj = get_etf_returns(etf, 'simple', 'Adj Close')
    df_ret_from_close = get_etf_returns(etf, 'simple', 'Close')
    # 2 simple div = ret Adj Close simple - ret Close simple
    df_div = df_ret_from_adj - df_ret_from_close
    # 3 convert to log if log
    if return_type=='log':
        df_div = np.log(df_div + 1)
    return df_div

def get_price_return(etf, return_type='log'):
    df_total = get_total_return(etf, 'simple')
    df_div = get_dividend_return(etf, 'simple')
    df_price = df_total - df_div
    if return_type == 'log':
        df_price = np.log(df_price + 1)
    return df_price

def get_joined_returns(d_weights, from_date=None, to_date=None):
    l_df = []
    for etf, value in d_weights.items():
        df_temp = get_total_return(etf, return_type='simple')
        l_df.append(df_temp)
    df_joined = pd.concat(l_df, axis=1)
    df_joined.sort_index(inplace=True)
    df_joined.dropna(inplace=True)
    fromdate = pd.to_datetime(from_date)
    todate = pd.to_datetime(to_date)
    filtered_df = df_joined.loc[fromdate:todate]
    return filtered_df

def get_portfolio_return(d_weights):
    df_joined = get_joined_returns(d_weights)
    df_weighted_returns = df_joined * pd.Series(d_weights)
    s_portfolio_return = df_weighted_returns.sum(axis=1)
    return pd.DataFrame(s_portfolio_return, columns=['pf'])

def get_portfolio_return_btw_dates(d_weights,
    from_date=None, to_date=None):
    df = get_portfolio_return(d_weights)
    fromdate = pd.to_datetime(from_date)
    todate = pd.to_datetime(to_date)
    filtered_df = df.loc[fromdate:todate]
    return filtered_df

df_portfolio_returns = get_portfolio_return_btw_dates(weights, '2007-04-16', '2022-12-20')

def subtract_trading_date(actual_date, x):
    date = pd.to_datetime(actual_date)
    # create a date range from the current date to `x` days ago
    date_range = pd.bdate_range(end=date, periods=x + 1)
    # subtract the last date in the range from the current date
    result = date_range[0]
    result_str = result.strftime('%Y-%m-%d')
    return result_str

def calc_simple_var(pf_value, d_weights, l_conf_levels,
    last_day_of_interval, window_in_days):
    from_date = subtract_trading_date(last_day_of_interval, window_in_days)
    df_ret = get_portfolio_return_btw_dates(
        d_weights, from_date, last_day_of_interval)
    l_quantiles = [1 - x for x in l_conf_levels]
    pf_mean = float(df_ret.mean())
    pf_std = float(df_ret.std())
    var_numbers = norm.ppf(l_quantiles, loc=pf_mean, scale=pf_std)
    df_result_ret = pd.DataFrame(var_numbers)
    df_result_ret.index = l_conf_levels
    df_result_ret = df_result_ret.transpose()
    df_result_ret.index = [last_day_of_interval]
    df_result_amount = df_result_ret * pf_value
    return df_result_ret, df_result_amount

def calc_covar_var(pf_value, d_weights, l_conf_levels,
    last_day_of_interval, window_in_days):
    from_date = subtract_trading_date(last_day_of_interval, window_in_days)
    df_rets = get_joined_returns(
        d_weights, from_date, last_day_of_interval)
    l_quantiles = [1 - x for x in l_conf_levels]
    means = df_rets.mean()
    covar = df_rets.cov()
    s_weights = pd.Series(d_weights)
    pf_mean = (s_weights * means).sum()
    pf_var = np.dot(s_weights.T, np.dot(covar, s_weights))
    var_numbers = norm.ppf(l_quantiles, loc=pf_mean, scale=np.sqrt(pf_var))
    df_result_ret = pd.DataFrame(var_numbers)
    df_result_ret.index = l_conf_levels
    df_result_ret = df_result_ret.transpose()
    df_result_ret.index = [last_day_of_interval]
    df_result_amount = df_result_ret * pf_value
    return df_result_ret, df_result_amount

def calc_historical_var(pf_value, d_weights, l_conf_levels,
    last_day_of_interval, window_in_days):
    from_date = subtract_trading_date(last_day_of_interval, window_in_days)
    df_ret = get_portfolio_return_btw_dates(
        d_weights, from_date, last_day_of_interval)
    l_quantiles = [1-x for x in l_conf_levels]
    df_result_ret = df_ret.quantile(l_quantiles)
    df_result_ret.index = l_conf_levels
    df_result_ret = df_result_ret.transpose()
    df_result_ret.index = [last_day_of_interval]
    df_result_amount = df_result_ret * pf_value
    return df_result_ret, df_result_amount

def calc_var_for_period(vartype,
    pf_value, d_weights, l_conf_levels,
    from_date, to_date,
    window_in_days):
    d_var_f = {
        'hist': calc_historical_var,
        'simple': calc_simple_var,
        'covar': calc_covar_var
    }
    f_var = d_var_f[vartype]
    business_days = pd.date_range(start=from_date, end=to_date, freq='B')
    df_result = None
    for last_day_of_interval in business_days:
        df_temp_, df_temp_amount = f_var(
            pf_value, d_weights, l_conf_levels,
            last_day_of_interval, window_in_days)
        if df_result is None:
            df_result = df_temp_amount
        else:
            df_result = pd.concat(
                [df_result, df_temp_amount],
                axis=0)
    return df_result


def calculate_historical_var(df_portfolio_returns, alpha):
    df_ret = df_portfolio_returns
    l_quantiles = [1-x for x in alpha]
    df_result_ret = df_ret.quantile(l_quantiles)
    df_result_ret.index = alpha
    df_result_ret = df_result_ret.transpose()
    #df_result_ret.index = ['2022-12-20']
    #df_result_amount = df_result_ret * 1000
    return df_result_ret #, df_result_amount

#df_returns = pd.DataFrame({'returns': np.arange(-0.05, 0.06, 0.01)})

#print(calculate_historical_var(df_returns, conf))

def calculate_covariance_matrix(volatility, corr):
    cov_xy = volatility[0] * volatility[1] * corr
    cov_matrix = np.array([[volatility[0] ** 2, cov_xy], [cov_xy, volatility[1] ** 2]])
    return cov_matrix

def calc_asset_returns(mean, cov, size):
    return np.random.multivariate_normal(mean, cov, size)


def simulated_returns(expected_return, volatility, correlation, numOfSim):
    corr = correlation
    covmat = calculate_covariance_matrix(volatility, corr)
    means = expected_return
    nsim = numOfSim
    rets = calc_asset_returns(means, covmat, nsim)
    return rets

returns_bnd = get_etf_returns('bnd','log','Adj Close')
returns_dba = get_etf_returns('dba','log','Adj Close')
df_returns = get_joined_returns(weights,'2007-04-10', '2022-12-20')
corr_matrix = df_returns.corr()

expected_return = [float(returns_bnd.mean(axis=0)), float(returns_dba.mean(axis=0))]
volatility = [float(returns_bnd.std(axis=0)), float(returns_dba.std(axis=0))]
correlation = corr_matrix.iloc[0,1]
numOfSim = 10000
print(simulated_returns(expected_return, volatility, correlation, numOfSim))
#print(returns_bnd.mean(axis=0))