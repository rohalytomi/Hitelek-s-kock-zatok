import pandas as pd
import numpy as np
import os
from scipy.stats import norm
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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



def calculate_ewma_variance(df_etf_returns, decay_factor, window):
    ewma = df_etf_returns.ewm(span=window, min_periods=window).var(decay_factor)
    return ewma

def create_data():
    window_size = 20
    filepath = "BND.csv"
    data = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    data['returns'] = np.log(data['Adj Close']).diff()
    data['squared_returns'] = data['returns'] ** 2
    cols = []
    for i in range(1, window_size + 1):
        col = f'lag_{i}'
        data[col] = data['squared_returns'].shift(i)
        cols.append(col)
    data.dropna(inplace=True)
    X = np.array(data[cols])
    y = np.array(data['squared_returns'])
    return X, y, data.index, data['squared_returns']

def create_polynomial_model(degree):
    model = LinearRegression()
    return f"Linear Regression (Degree {degree})", model

def train_and_evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    return model, mse, y_pred

def print_coeffs(label, model):
    print(label, model.coef_, model.intercept_)

def plot_cross_validation_error(results):
    degrees = list(results.keys())
    mse_values = list(results.values())

    plt.plot(degrees, mse_values, marker='o')
    plt.xlabel('Degree')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Cross-Validation Error')
    plt.grid(True)
    plt.show()

def plot_actual_vs_predicted(index, y_actual, y_pred):
    plt.plot(index, y_actual[:len(y_pred)], label='Actual')
    plt.plot(index, y_pred, label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Squared Returns')
    plt.title('Actual vs. Predicted Squared Returns')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

def plot_lagged_squared_returns(X, y):
    plt.scatter(X[:, -1], y, alpha=0.5)
    plt.xlabel('Lagged Squared Returns')
    plt.ylabel('Squared Returns')
    plt.title('Lagged Squared Returns vs. Actual Squared Returns')
    plt.grid(True)
    plt.show()

def cross_validate(X, y, n_splits=5, from_degree=1, to_degree=10):
    degrees = range(from_degree, to_degree+1)
    kf = KFold(n_splits=n_splits)
    results = {}
    best_model = None
    best_degree = None
    best_mse = np.inf
    np.set_printoptions(precision=4)
    for degree in degrees:
        name, model = create_polynomial_model(degree)
        mse_sum = 0
        y_actual = np.array([])
        y_pred_all = np.array([])
        for train_idx, val_idx in kf.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            model, mse, y_pred = train_and_evaluate_model(model, X_train, y_train, X_val, y_val)
            mse_sum += mse
            y_actual = np.concatenate((y_actual, y_val))
            y_pred_all = np.concatenate((y_pred_all, y_pred))
        avg_mse = mse_sum / n_splits
        results[degree] = avg_mse
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_degree = degree
            best_model = model
            best_y_pred = y_pred_all

    print(f"Best model: degree={best_degree}, MSE={best_mse}")
    print_coeffs("Coefficients for best model: ", best_model)

    plot_cross_validation_error(results)
    plot_actual_vs_predicted(index, y_actual, best_y_pred)
    plot_lagged_squared_returns(X, y)

    return best_model

def hyperparameter_search(X_train, X_test, y_train, y_test, from_degree=1, to_degree=15):
    degrees = range(from_degree, to_degree+1)
    best_degree, best_mse, best_model = None, float('inf'), None
    d_mse = {}
    for degree in degrees:
        name, model = create_polynomial_model(degree)
        model, mse, y_pred = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
        d_mse[degree] = mse
        if mse < best_mse:
            best_degree, best_mse, best_model = degree, mse, model
    print(f'Best degree: {best_degree}, Best MSE {best_mse}')
    print_coeffs('Coefficients: ', best_model)
    return best_model

