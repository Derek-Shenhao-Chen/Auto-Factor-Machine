import pandas as pd 
import numpy as np
from numba import njit, prange
import warnings
warnings.filterwarnings("ignore")


"""
Define efficient operator functions for expression calculation.
"""



# Simple operator
def abs(x):
    return np.abs(x)

def inv(x):
    return pd.DataFrame(
        np.where(np.abs(x) > 0, 1/x, np.nan),
        index=x.index, columns=x.columns)

def neg(x):
    return -x

def ln(x):
    return pd.DataFrame(
        np.where(x > 0, np.log(x), np.nan),
        index=x.index, columns=x.columns)

def exp_e(x: pd.DataFrame):
    return pd.DataFrame(np.power(np.e, x), index=x.index, columns=x.columns)

def sign(x):
    return np.sign(x)

def square(x):
    return np.square(x)

def sign_square(x):
    return sign(x) * square(x)

def power_e(x):
    return np.power(x,np.e)

def sqrt(x):
    return pd.DataFrame(
        np.where(x >= 0, np.sqrt(x), np.nan),
        index=x.index, columns=x.columns)

def sign_sqrt(x):
    return pd.DataFrame(
        np.where(x >= 0, np.sqrt(x), - np.sqrt(abs(x))),
        index=x.index, columns=x.columns)

def demean(x):
    raw_arr = x.to_numpy(dtype=np.float64, na_value=np.nan)
    mean = np.nanmean(raw_arr,axis=1).reshape(-1, 1)
    result = np.abs(raw_arr - mean)
    return pd.DataFrame(result, index=x.index, columns=x.columns)

def demedian(x):
    raw_arr = x.to_numpy(dtype=np.float64, na_value=np.nan)
    median = np.nanmedian(raw_arr,axis=1).reshape(-1, 1)
    result = np.abs(raw_arr - median)
    return pd.DataFrame(result, index=x.index, columns=x.columns)

def max_val(x, y):
    return pd.DataFrame(np.maximum(x.values, y.values),index=x.index, columns=x.columns)

def min_val(x, y):
    return pd.DataFrame(np.minimum(x.values, y.values),index=x.index, columns=x.columns)



# Logic operator
def is_larger(x, y):
    return x > y

def is_smaller(x, y):
    return x < y




# Cross-section operator
def cs_rank(x):
    raw_arr = x.to_numpy(dtype=np.float64, na_value=np.nan)
    nan_mask = np.isnan(raw_arr)
    result = np.full_like(raw_arr, np.nan, dtype=np.float64)
    for i in range(raw_arr.shape[0]):
        row = raw_arr[i]
        mask = ~nan_mask[i]
        valid_values = row[mask]
        if len(valid_values) < 2:
            continue
        sorted_indices = np.argsort(valid_values)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(valid_values)) + 1
        result[i,mask] = ranks / len(valid_values)
    return pd.DataFrame(result, index=x.index, columns=x.columns)

def cs_mean(x):
    raw_arr = x.to_numpy(dtype=np.float64, na_value=np.nan)
    n,m = raw_arr.shape
    result = np.nanmean(raw_arr,axis=1)
    result_2d = np.tile(result.reshape(n, 1), (1, m))
    return pd.DataFrame(result_2d, index=x.index, columns=x.columns)

def cs_median(x):
    raw_arr = x.to_numpy(dtype=np.float64, na_value=np.nan)
    n,m = raw_arr.shape
    result = np.nanmedian(raw_arr,axis=1)
    result_2d = np.tile(result.reshape(n, 1), (1, m))
    return pd.DataFrame(result_2d, index=x.index, columns=x.columns)

def cs_sum(x):
    raw_arr = x.to_numpy(dtype=np.float64, na_value=np.nan)
    n,m = raw_arr.shape
    result = np.nansum(raw_arr,axis=1)
    result_2d = np.tile(result.reshape(n, 1), (1, m))
    return pd.DataFrame(result_2d, index=x.index, columns=x.columns)

def cs_std(x):
    raw_arr = x.to_numpy(dtype=np.float64, na_value=np.nan)
    n,m = raw_arr.shape
    result = np.nanstd(raw_arr,axis=1)
    result_2d = np.tile(result.reshape(n, 1), (1, m))
    return pd.DataFrame(result_2d, index=x.index, columns=x.columns)

def cs_corr(x,y):
    raw_arr_x = x.to_numpy(dtype=np.float64, na_value=np.nan)
    raw_arr_y = y.to_numpy(dtype=np.float64, na_value=np.nan)
    demean_x = raw_arr_x - np.nanmean(raw_arr_x,axis=1).reshape(-1,1)
    demean_y = raw_arr_y - np.nanmean(raw_arr_y,axis=1).reshape(-1,1)
    numerator = np.nansum(demean_x * demean_y, axis=1)
    std_x = np.sqrt(np.nansum(demean_x **2, axis=1))
    std_y = np.sqrt(np.nansum(demean_y** 2, axis=1))
    denominator = std_x * std_y
    denominator[denominator < 0] = np.nan
    corr = numerator / denominator
    n,m = raw_arr_x.shape
    result = np.tile(corr.reshape(n, 1), (1, m))
    return pd.DataFrame(result, index=x.index, columns=x.columns)

def cs_cov(x,y):
    raw_arr_x = x.to_numpy(dtype=np.float64, na_value=np.nan)
    raw_arr_y = y.to_numpy(dtype=np.float64, na_value=np.nan)
    demean_x = raw_arr_x - np.nanmean(raw_arr_x,axis=1).reshape(-1,1)
    demean_y = raw_arr_y - np.nanmean(raw_arr_y,axis=1).reshape(-1,1)
    cov = np.nanmean(demean_x * demean_y, axis=1)
    n,m = raw_arr_x.shape
    result = np.tile(cov.reshape(n, 1), (1, m))
    return pd.DataFrame(result, index=x.index, columns=x.columns)

def cs_ols_beta(y,x):
    raw_arr_x = x.to_numpy(dtype=np.float64, na_value=np.nan)
    raw_arr_y = y.to_numpy(dtype=np.float64, na_value=np.nan)
    demean_x = raw_arr_x - np.nanmean(raw_arr_x,axis=1).reshape(-1,1)
    demean_y = raw_arr_y - np.nanmean(raw_arr_y,axis=1).reshape(-1,1)
    cov = np.nanmean(demean_x * demean_y, axis=1)
    var = np.nanmean(demean_x**2,axis=1)
    beta = np.where(var > 0, cov / var, np.nan)
    n,m = raw_arr_x.shape
    result = np.tile(beta.reshape(n, 1), (1, m))
    return pd.DataFrame(result, index=x.index, columns=x.columns)

def cs_ols_resid(y,x):
    raw_arr_x = x.to_numpy(dtype=np.float64, na_value=np.nan)
    raw_arr_y = y.to_numpy(dtype=np.float64, na_value=np.nan)
    demean_x = raw_arr_x - np.nanmean(raw_arr_x,axis=1).reshape(-1,1)
    demean_y = raw_arr_y - np.nanmean(raw_arr_y,axis=1).reshape(-1,1)
    cov = np.nanmean(demean_x * demean_y, axis=1)
    var = np.nanmean(demean_x**2,axis=1)
    beta = np.where(var > 0, cov / var, np.nan).reshape(-1,1)
    y_pred = beta * raw_arr_x
    resid = raw_arr_y - y_pred
    intercept = np.nanmean(resid, axis=1).reshape(-1,1)
    result = resid - intercept
    return pd.DataFrame(result, index=x.index, columns=x.columns)

def cs_ols_intercept(y,x):
    raw_arr_x = x.to_numpy(dtype=np.float64, na_value=np.nan)
    raw_arr_y = y.to_numpy(dtype=np.float64, na_value=np.nan)
    demean_x = raw_arr_x - np.nanmean(raw_arr_x,axis=1).reshape(-1,1)
    demean_y = raw_arr_y - np.nanmean(raw_arr_y,axis=1).reshape(-1,1)
    cov = np.nanmean(demean_x * demean_y, axis=1)
    var = np.nanmean(demean_x**2,axis=1)
    beta = np.where(var > 0, cov / var, np.nan).reshape(-1,1)
    y_pred = beta * raw_arr_x
    resid = raw_arr_y - y_pred
    intercept = np.nanmean(resid, axis=1).reshape(-1,1)
    n,m = raw_arr_x.shape
    result = np.tile(intercept.reshape(n, 1), (1, m))
    return pd.DataFrame(result, index=x.index, columns=x.columns)



# Time-series operator
def shift(x,d=1):
    return x.shift(d)

def shift_one(x):
    return x.shift()

def pct_change(x,d=1):
    return x.pct_change(periods=d, fill_method=None)

def pct_change_one(x):
    return x.pct_change(fill_method=None)

def diff(x,d=1):
    return x.diff(d)

def diff_one(x):
    return x.diff()

def ts_mean(x,d):
    raw_arr = x.to_numpy(dtype=np.float64, na_value=np.nan)
    result = np.full_like(raw_arr, np.nan, dtype=np.float64)
    for i in range(d-1,raw_arr.shape[0]):
        window_data = raw_arr[i-d+1:i+1,:]
        result[i,:] = np.nanmean(window_data,axis=0)
    return pd.DataFrame(result, index=x.index, columns=x.columns)

def ts_sum(x,d):
    raw_arr = x.to_numpy(dtype=np.float64, na_value=np.nan)
    result = np.full_like(raw_arr, np.nan, dtype=np.float64)
    for i in range(d-1,raw_arr.shape[0]):
        window_data = raw_arr[i-d+1:i+1,:]
        result[i,:] = np.nansum(window_data,axis=0)
    return pd.DataFrame(result, index=x.index, columns=x.columns)

def ts_prod(x,d):
    raw_arr = x.to_numpy(dtype=np.float64, na_value=np.nan)
    result = np.full_like(raw_arr, np.nan, dtype=np.float64)
    for i in range(d-1,raw_arr.shape[0]):
        window_data = raw_arr[i-d+1:i+1,:]
        result[i,:] = np.nanprod(window_data,axis=0)
    return pd.DataFrame(result, index=x.index, columns=x.columns)

def ts_std(x,d):
    raw_arr = x.to_numpy(dtype=np.float64, na_value=np.nan)
    result = np.full_like(raw_arr, np.nan, dtype=np.float64)
    for i in range(d-1,raw_arr.shape[0]):
        window_data = raw_arr[i-d+1:i+1,:]
        result[i,:] = np.nanstd(window_data,axis=0)
    return pd.DataFrame(result, index=x.index, columns=x.columns)

def ts_max(x,d):
    raw_arr = x.to_numpy(dtype=np.float64, na_value=np.nan)
    result = np.full_like(raw_arr, np.nan, dtype=np.float64)
    for i in range(d-1,raw_arr.shape[0]):
        window_data = raw_arr[i-d+1:i+1,:]
        result[i,:] = np.nanmax(window_data,axis=0)
    return pd.DataFrame(result, index=x.index, columns=x.columns)

def ts_min(x,d):
    raw_arr = x.to_numpy(dtype=np.float64, na_value=np.nan)
    result = np.full_like(raw_arr, np.nan, dtype=np.float64)
    for i in range(d-1,raw_arr.shape[0]):
        window_data = raw_arr[i-d+1:i+1,:]
        result[i,:] = np.nanmin(window_data,axis=0)
    return pd.DataFrame(result, index=x.index, columns=x.columns)

def ts_argmax(x,d):
    raw_arr = x.to_numpy(dtype=np.float64, na_value=np.nan)
    result = np.full_like(raw_arr, np.nan, dtype=np.float64)
    for i in range(d-1,raw_arr.shape[0]):
        window_data = raw_arr[i-d+1:i+1,:]
        is_all_nan = np.all(np.isnan(window_data), axis=0)
        sort_idx = np.full(window_data.shape[1], np.nan)
        non_nan_cols = ~is_all_nan
        if np.any(non_nan_cols):
            sort_idx[non_nan_cols] = np.nanargmax(window_data[:, non_nan_cols], axis=0)
        result[i, :] = np.where(is_all_nan, np.nan, d - sort_idx - 1)
    return pd.DataFrame(result, index=x.index, columns=x.columns)

def ts_argmin(x,d):
    raw_arr = x.to_numpy(dtype=np.float64, na_value=np.nan)
    result = np.full_like(raw_arr, np.nan, dtype=np.float64)
    for i in range(d-1,raw_arr.shape[0]):
        window_data = raw_arr[i-d+1:i+1,:]
        is_all_nan = np.all(np.isnan(window_data), axis=0)
        sort_idx = np.full(window_data.shape[1], np.nan)
        non_nan_cols = ~is_all_nan
        if np.any(non_nan_cols):
            sort_idx[non_nan_cols] = np.nanargmin(window_data[:, non_nan_cols], axis=0)
        result[i, :] = np.where(is_all_nan, np.nan, d - sort_idx - 1)
    return pd.DataFrame(result, index=x.index, columns=x.columns)

def ts_rank(x,d):
    raw_arr = x.to_numpy(dtype=np.float64, na_value=np.nan)
    result = np.full_like(raw_arr, np.nan, dtype=np.float64)
    for col in range(raw_arr.shape[1]):
        result[:,col] = _rolling_rank_numba(raw_arr[:,col], d)
    return pd.DataFrame(result, index=x.index, columns=x.columns)

def ts_corr(x,y,d):
    raw_arr_x = x.to_numpy(dtype=np.float64, na_value=np.nan)
    raw_arr_y = y.to_numpy(dtype=np.float64, na_value=np.nan)
    result = np.full_like(raw_arr_x, np.nan, dtype=np.float64)
    for i in range(d-1,raw_arr_x.shape[0]):
        window_x = raw_arr_x[i-d+1:i+1,:]
        window_y = raw_arr_y[i-d+1:i+1,:]
        demean_x = window_x - np.nanmean(window_x,axis=0)
        demean_y = window_y - np.nanmean(window_y,axis=0)
        numerator = np.nansum(demean_x * demean_y, axis=0)
        std_x = np.sqrt(np.nansum(demean_x **2, axis=0))
        std_y = np.sqrt(np.nansum(demean_y** 2, axis=0))
        denominator = std_x * std_y
        denominator[denominator < 0] = np.nan
        corr = numerator / denominator
        result[i,:] = corr
    return pd.DataFrame(result, index=x.index, columns=x.columns)

def ts_cov(x,y,d):
    raw_arr_x = x.to_numpy(dtype=np.float64, na_value=np.nan)
    raw_arr_y = y.to_numpy(dtype=np.float64, na_value=np.nan)
    result = np.full_like(raw_arr_x, np.nan, dtype=np.float64)
    for i in range(d-1,raw_arr_x.shape[0]):
        window_x = raw_arr_x[i-d+1:i+1,:]
        window_y = raw_arr_y[i-d+1:i+1,:]
        demean_x = window_x - np.nanmean(window_x,axis=0)
        demean_y = window_y - np.nanmean(window_y,axis=0)
        cov = np.nanmean(demean_x * demean_y, axis=0)
        result[i,:] = cov
    return pd.DataFrame(result, index=x.index, columns=x.columns)

def ts_ols_beta(y,x,d):
    raw_arr_x = x.to_numpy(dtype=np.float64, na_value=np.nan)
    raw_arr_y = y.to_numpy(dtype=np.float64, na_value=np.nan)
    result = np.full_like(raw_arr_x, np.nan, dtype=np.float64)
    for i in range(d-1,raw_arr_x.shape[0]):
        window_x = raw_arr_x[i-d+1:i+1,:]
        window_y = raw_arr_y[i-d+1:i+1,:]
        demean_x = window_x - np.nanmean(window_x,axis=0)
        demean_y = window_y - np.nanmean(window_y,axis=0)
        cov = np.nanmean(demean_x * demean_y, axis=0)
        var = np.nanmean(demean_x**2, axis=0)
        beta = np.where(var > 0, cov / var, np.nan)
        result[i,:] = beta
    return pd.DataFrame(result, index=x.index, columns=x.columns)

def ts_ols_intercept(y,x,d):
    raw_arr_x = x.to_numpy(dtype=np.float64, na_value=np.nan)
    raw_arr_y = y.to_numpy(dtype=np.float64, na_value=np.nan)
    result = np.full_like(raw_arr_x, np.nan, dtype=np.float64)
    for i in range(d-1,raw_arr_x.shape[0]):
        window_x = raw_arr_x[i-d+1:i+1,:]
        window_y = raw_arr_y[i-d+1:i+1,:]
        demean_x = window_x - np.nanmean(window_x,axis=0)
        demean_y = window_y - np.nanmean(window_y,axis=0)
        cov = np.nanmean(demean_x * demean_y, axis=0)
        var = np.nanmean(demean_x**2, axis=0)
        beta = np.where(var > 0, cov / var, np.nan).reshape(1,-1)
        resid = window_y - beta * window_x
        intercept = np.nanmean(resid,axis=0)
        result[i,:] = intercept
    return pd.DataFrame(result, index=x.index, columns=x.columns)

def ts_ols_resid_std(y,x,d):
    raw_arr_x = x.to_numpy(dtype=np.float64, na_value=np.nan)
    raw_arr_y = y.to_numpy(dtype=np.float64, na_value=np.nan)
    result = np.full_like(raw_arr_x, np.nan, dtype=np.float64)
    for i in range(d-1,raw_arr_x.shape[0]):
        window_x = raw_arr_x[i-d+1:i+1,:]
        window_y = raw_arr_y[i-d+1:i+1,:]
        demean_x = window_x - np.nanmean(window_x,axis=0)
        demean_y = window_y - np.nanmean(window_y,axis=0)
        cov = np.nanmean(demean_x * demean_y, axis=0)
        var = np.nanmean(demean_x**2, axis=0)
        beta = np.where(var > 0, cov / var, np.nan).reshape(1,-1)
        resid = window_y - beta * window_x
        intercept = np.nanmean(resid,axis=0).reshape(1,-1)
        resid = resid - intercept
        resid_std = np.nanstd(resid,axis=0)
        result[i,:] = resid_std
    return pd.DataFrame(result, index=x.index, columns=x.columns)

def ts_trend_beta(y,d):
    raw_arr_y = y.to_numpy(dtype=np.float64, na_value=np.nan)
    series_x = np.arange(1, d+1).reshape(-1, 1)
    demean_x = series_x - np.nanmean(series_x,axis=0)
    result = np.full_like(raw_arr_y, np.nan, dtype=np.float64)
    for i in range(d-1,raw_arr_y.shape[0]):
        window_y = raw_arr_y[i-d+1:i+1,:]
        demean_y = window_y - np.nanmean(window_y,axis=0)
        cov = np.nanmean(demean_x * demean_y, axis=0)
        var = np.nanmean(demean_x**2, axis=0)
        beta = np.where(var > 0, cov / var, np.nan)
        result[i,:] = beta
    return pd.DataFrame(result, index=y.index, columns=y.columns)

def ts_trend_intercept(y,d):
    raw_arr_y = y.to_numpy(dtype=np.float64, na_value=np.nan)
    series_x = np.arange(1, d+1).reshape(-1, 1)
    demean_x = series_x - np.nanmean(series_x,axis=0)
    result = np.full_like(raw_arr_y, np.nan, dtype=np.float64)
    for i in range(d-1,raw_arr_y.shape[0]):
        window_y = raw_arr_y[i-d+1:i+1,:]
        demean_y = window_y - np.nanmean(window_y,axis=0)
        cov = np.nanmean(demean_x * demean_y, axis=0)
        var = np.nanmean(demean_x**2, axis=0)
        beta = np.where(var > 0, cov / var, np.nan).reshape(1,-1)
        resid = window_y - beta * series_x
        intercept = np.nanmean(resid,axis=0)
        result[i,:] = intercept
    return pd.DataFrame(result, index=y.index, columns=y.columns)

def ts_trend_resid_std(y,d):
    raw_arr_y = y.to_numpy(dtype=np.float64, na_value=np.nan)
    series_x = np.arange(1, d+1).reshape(-1, 1)
    demean_x = series_x - np.nanmean(series_x,axis=0)
    result = np.full_like(raw_arr_y, np.nan, dtype=np.float64)
    for i in range(d-1,raw_arr_y.shape[0]):
        window_y = raw_arr_y[i-d+1:i+1,:]
        demean_y = window_y - np.nanmean(window_y,axis=0)
        cov = np.nanmean(demean_x * demean_y, axis=0)
        var = np.nanmean(demean_x**2, axis=0)
        beta = np.where(var > 0, cov / var, np.nan).reshape(1,-1)
        resid = window_y - beta * series_x
        intercept = np.nanmean(resid,axis=0).reshape(1,-1)
        resid = resid - intercept
        resid_std = np.nanstd(resid,axis=0)
        result[i,:] = resid_std
    return pd.DataFrame(result, index=y.index, columns=y.columns)



# Conditional operator
def where(condition, x, y):
    cond_arr = condition.to_numpy(dtype=bool, na_value=np.nan)
    x_arr = x.to_numpy(dtype=np.float64, na_value=np.nan) 
    y_arr = y.to_numpy(dtype=np.float64, na_value=np.nan) 
    result_arr = np.where(cond_arr, x_arr, y_arr)
    return pd.DataFrame(result_arr, index=condition.index, columns=condition.columns)


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _rolling_rank_numba(arr, window):
    n = len(arr)
    result = np.full(n, np.nan)
    for i in prange(window-1, n):
        window_data = arr[i-window+1:i+1]
        if np.isnan(window_data[-1]):
            continue
        mask = ~np.isnan(window_data)
        valid_data = window_data[mask]
        if len(valid_data) < 2:
            continue
        sorted_indices = np.argsort(valid_data)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(valid_data)) + 1
        result[i] = ranks[-1] / len(valid_data)
    return result