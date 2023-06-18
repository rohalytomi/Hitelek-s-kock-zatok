import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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

# Main code
X, y, index, squared_returns = create_data()
best_model = cross_validate(X, y, n_splits=5, from_degree=1, to_degree=20)
