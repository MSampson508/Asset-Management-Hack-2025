import datetime
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    # for timing purpose
    print(datetime.datetime.now())

    # turn off pandas Setting with Copy Warning
    pd.set_option("mode.chained_assignment", None)

    # set working directory
    work_dir = r"/teamspace/studios/this_studio/"

    '''
    # read sample data
    file_path = os.path.join(
        work_dir, "retsampletrimmed.csv"
    )  # replace with the correct file name
    raw = pd.read_csv(
        file_path, parse_dates=["ret_eom"], low_memory=False
    )  # the date is the first day of the return month (t+1)

    # read list of predictors for stocks
    file_path = os.path.join(
        work_dir, "mkt_ind.csv"
    )  # replace with the correct file name
    stock_vars = list(pd.read_csv(file_path)["variable"].values)

    # define the left hand side variable
    ret_var = "stock_ret"
    new_set = raw[
        raw[ret_var].notna()
    ].copy()  # create a copy of the data and make sure the left hand side is not missing

    # transform each variable in each month to the same scale
    monthly = new_set.groupby("date")
    data = pd.DataFrame()
    for date, monthly_raw in monthly:
        group = monthly_raw.copy()
        # rank transform each variable to [-1, 1]
        for var in stock_vars:
            var_median = group[var].median(skipna=True)
            group[var] = group[var].fillna(
                var_median
            )  # fill missing values with the cross-sectional median of each month

            group[var] = group[var].rank(method="dense") - 1
            group_max = group[var].max()
            if group_max > 0:
                group[var] = (group[var] / group_max) * 2 - 1
            else:
                group[var] = 0  # in case of all missing values
                print("Warning:", date, var, "set to zero.")

        # add the adjusted values
        data = data._append(
            group, ignore_index=True
        )  # append may not work with certain versions of pandas, use concat instead if needed

    # initialize the starting date, counter, and output data
    starting = pd.to_datetime("20050101", format="%Y%m%d")
    counter = 0
    pred_out = pd.DataFrame()
'''

    #MY code starts here
    preds = pd.read_csv(os.path.join(work_dir, "preds.csv"))
    preds["ret_eom"] = pd.to_datetime(preds["ym"], format="%Y-%m")
    preds["id"] = preds["gvkey"].astype(str)
    preds = preds.rename(columns={"r_hat": "my_model"})

    raw = pd.read_csv(
        os.path.join(work_dir, "retsampletrimmed.csv"),
        parse_dates=["ret_eom"],
        low_memory=False
    )
        # ensure we have an 'id' column to merge on
    if "id" not in raw.columns and "gvkey" in raw.columns:
        raw = raw.rename(columns={"gvkey": "id"})
    raw["id"] = raw["id"].astype(str)


    eval_cols = ["ret_eom", "id", "stock_ret"]   # ret_var == "stock_ret"
    panel = raw[eval_cols].copy()
    panel["id"] = panel["id"].astype(str)

    
    pred_out = panel.merge(
        preds[["ret_eom", "id", "my_model"]],
        on=["ret_eom", "id"],
        how="inner"           # use inner to ensure aligned rows
    )

    yreal = pred_out["stock_ret"].astype("float64").to_numpy()
    ypred = pred_out["my_model"].astype("float64").to_numpy()
    r2 = 1 - np.nansum((yreal - ypred)**2) / np.nansum(yreal**2)
    print("my_model R2:", r2)

    # 2) Build benchmark series from mkt_ind.csv  (monthly market returns)
    mkt = pd.read_csv(os.path.join(work_dir, "mkt_ind.csv"))
    mkt["ret_eom"] = pd.to_datetime(
        mkt["year"].astype(int).astype(str) + "-" + mkt["month"].astype(int).astype(str).str.zfill(2)
    )
    spx = mkt.set_index("ret_eom")["ret"].astype(float)  

    def long_short(df, score_col="my_model", q=0.10):
        def one_month(x):
            k = max(1, int(len(x)*q))
            long  = x.nlargest(k, score_col)["stock_ret"].mean()
            short = x.nsmallest(k, score_col)["stock_ret"].mean()
            return long - short
        return df.groupby("ret_eom", sort=True).apply(one_month).rename("port_ret")

    port = long_short(pred_out)


    idx = port.index.intersection(spx.index)
    p = port.loc[idx].astype(float)
    b = spx.loc[idx].astype(float)



    ann_ret = p.mean()*12
    ann_vol = p.std(ddof=1)*np.sqrt(12)
    sharpe  = (p.mean()/p.std(ddof=1))*np.sqrt(12)
    active  = p - b
    info    = (active.mean()*12) / (active.std(ddof=1)*np.sqrt(12))
    # CAPM alpha (monthly OLS, annualized)
    X = np.c_[np.ones(len(idx)), b.values]
    alpha_m, beta = np.linalg.lstsq(X, p.values, rcond=None)[0]
    alpha_ann = alpha_m*12
    wealth = (1+p).cumprod()
    max_dd = (wealth/wealth.cummax() - 1).min()

    print(f"Portfolio vs Market — ann_ret:{ann_ret:.6f} ann_vol:{ann_vol:.6f} "
      f"Sharpe:{sharpe:.3f} Info:{info:.3f} Alpha_ann:{alpha_ann:.6f} MaxDD:{max_dd:.3f}")

#My code ends here



    # output the predicted value to csv
    out_path = os.path.join(work_dir, "output.csv")
    print(out_path)
    pred_out.to_csv(out_path, index=False)



    # for timing purpose
    print(datetime.datetime.now())



'''
    # estimation with expanding window
    while (starting + pd.DateOffset(years=11 + counter)) <= pd.to_datetime(
        "20260101", format="%Y%m%d"
    ):
        cutoff = [
            starting,
            starting
            + pd.DateOffset(
                years=8 + counter
            ),  # use 8 years and expanding as the training set
            starting
            + pd.DateOffset(
                years=10 + counter
            ),  # use the next 2 years as the validation set
            starting + pd.DateOffset(years=11 + counter),
        ]  # use the next year as the out-of-sample testing set

        # cut the sample into training, validation, and testing sets
        train = data[(data["date"] >= cutoff[0]) & (data["date"] < cutoff[1])]
        validate = data[(data["date"] >= cutoff[1]) & (data["date"] < cutoff[2])]
        test = data[(data["date"] >= cutoff[2]) & (data["date"] < cutoff[3])]

        # Optional: if your data has additional binary or categorical variables,
        # you can further standardize them here
        scaler = StandardScaler().fit(train[stock_vars])
        train[stock_vars] = scaler.transform(train[stock_vars])
        validate[stock_vars] = scaler.transform(validate[stock_vars])
        test[stock_vars] = scaler.transform(test[stock_vars])

        # get Xs and Ys
        X_train = train[stock_vars].values
        Y_train = train[ret_var].values
        X_val = validate[stock_vars].values
        Y_val = validate[ret_var].values
        X_test = test[stock_vars].values
        Y_test = test[ret_var].values

        # de-mean Y (because the regressions are fitted without an intercept)
        # if you want to include an intercept (or bias in neural networks, etc), you can skip this step
        Y_mean = np.mean(Y_train)
        Y_train_dm = Y_train - Y_mean

        # prepare output data
        reg_pred = test[
            ["year", "month", "ret_eom", "id", ret_var]
        ]  # minimum identifications for each stock

        # Linear Regression
        # no validation is needed for OLS
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_train, Y_train_dm)
        x_pred = reg.predict(X_test) + Y_mean
        reg_pred["ols"] = x_pred

        # Lasso
        lambdas = np.arange(
            -4, 4.1, 0.1
        )  # search for the best lambda in the range of 10^-4 to 10^4, range can be adjusted
        val_mse = np.zeros(len(lambdas))
        for ind, i in enumerate(lambdas):
            reg = Lasso(alpha=(10**i), max_iter=1000000, fit_intercept=False)
            reg.fit(X_train, Y_train_dm)
            val_mse[ind] = mean_squared_error(Y_val, reg.predict(X_val) + Y_mean)

        # select the best lambda based on the validation set
        best_lambda = lambdas[np.argmin(val_mse)]
        reg = Lasso(alpha=(10**best_lambda), max_iter=1000000, fit_intercept=False)
        reg.fit(X_train, Y_train_dm)
        x_pred = reg.predict(X_test) + Y_mean  # predict the out-of-sample testing set
        reg_pred["lasso"] = x_pred

        # Ridge
        # same format as above
        lambdas = np.arange(-1, 8.1, 0.1)
        val_mse = np.zeros(len(lambdas))
        for ind, i in enumerate(lambdas):
            reg = Ridge(alpha=((10**i) * 0.5), fit_intercept=False)
            reg.fit(X_train, Y_train_dm)
            val_mse[ind] = mean_squared_error(Y_val, reg.predict(X_val) + Y_mean)

        best_lambda = lambdas[np.argmin(val_mse)]
        reg = Ridge(alpha=((10**best_lambda) * 0.5), fit_intercept=False)
        reg.fit(X_train, Y_train_dm)
        x_pred = reg.predict(X_test) + Y_mean
        reg_pred["ridge"] = x_pred

        # Elastic Net
        # same format as above
        lambdas = np.arange(-4, 4.1, 0.1)
        val_mse = np.zeros(len(lambdas))
        for ind, i in enumerate(lambdas):
            reg = ElasticNet(alpha=(10**i), max_iter=1000000, fit_intercept=False)
            reg.fit(X_train, Y_train_dm)
            val_mse[ind] = mean_squared_error(Y_val, reg.predict(X_val) + Y_mean)

        best_lambda = lambdas[np.argmin(val_mse)]
        reg = ElasticNet(alpha=(10**best_lambda), max_iter=1000000, fit_intercept=False)
        reg.fit(X_train, Y_train_dm)
        x_pred = reg.predict(X_test) + Y_mean
        reg_pred["en"] = x_pred

        # add to the output data
        pred_out = pred_out._append(reg_pred, ignore_index=True)

        # go to the next year
        counter += 1
'''