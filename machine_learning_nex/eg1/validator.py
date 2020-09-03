import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, roc_auc_score, mean_squared_log_error


def data_preprocess():
    df = pd.read_csv("cs-1.csv", index_col=0)
    df_unknows = df[df.MonthlyIncome.isnull()]
    df_knows = df[df.MonthlyIncome.notnull()]
    df_knows = df_knows[df_knows["MonthlyIncome"] > 1000]

    # month_income_anomaly_detect(df_knows)

    x_train = df_knows.iloc[:100000, 1:].values
    y_train = (df_knows[df_knows.columns[0]]).values[:100000]

    x_test = df_knows.iloc[100000:, 1:].values
    y_test = df_knows[df_knows.columns[0]].values[100000:]
    return x_train, y_train, x_test, y_test


def tune_parameters(x_train, y_train):
    n_estimator = list(np.arange(5, 30, 5))
    max_depth = list(np.arange(5, 50, 5))
    random_grid = {
        "n_estimators": n_estimator,
        "max_depth": max_depth
    }

    rfr = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator=rfr, param_distributions=random_grid, n_iter=30, cv=3, n_jobs=-1,
                                   verbose=2)
    rf_random.fit(x_train, y_train)

    print(rf_random.best_params_)


def evaluate(model, x_test, y_test, x_train, y_train):
    predictions = model.predict(x_test)
    print(predictions[:10])
    print(y_test[:10])
    print(len(x_test))
    avg_error = mean_squared_log_error(y_test, predictions)
    print(" mean_squared_log_errors: ", avg_error)


def evaluate_params(x_train, y_train, x_test, y_test):
    rfr = RandomForestRegressor(n_estimators=15, max_depth=10)
    rfr.fit(x_train, y_train)
    evaluate(rfr, x_test, y_test, x_train, y_train)


def main():
    x_train, y_train, x_test, y_test = data_preprocess()
    # tune_parameters(x_train, y_train)
    evaluate_params(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
