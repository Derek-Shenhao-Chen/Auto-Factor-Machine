import pandas as pd
import numpy as np
import os

from tools.FunctionSet import *
from tools.DataLoader import FullSample, SmallSample



class ExpressionTest:
    """
    ExpressionTest class evaluates generated expressions against financial data
    to determine their effectiveness as predictive factors. It performs a two-stage
    validation process and saves only those expressions that pass all tests.
    """

    def __init__(self, factor_data_base: str):
        self.path = factor_data_base


    def run(self, factor_name: str):
        """
        Calculate and test an expression, storing results for expressions that pass validation.
        The method performs a two-round testing process using different data samples.
        """
        _, result = self._calculate_and_test(factor_name, 1)
        if result == 'pass':
            factor, result = self._calculate_and_test(factor_name, 2)
            if result == 'pass':
                self._save(factor, factor_name)


    def _calculate_and_test(self, factor_name: str, round_num: int):
        """
        Calculate and test an expression using specified data sample.
        """

        # Select corresponding dataset based on round number
        variables = ['ret_df', 'turn_df', 'amount_df', 'volume_df', 'marketcap_df', 'excess_df', 'market_df', 
                               'open_df', 'high_df', 'low_df', 'close_df']
        if round_num == 1:
            datasample = 'SmallSample'
            y_label = SmallSample.y_label
        elif round_num ==2:
            datasample = 'FullSample'
            y_label = FullSample.y_label
        
        mapping = {}
        for i in variables:
            mapping[i] = datasample + '.' + i
            
        name = factor_name
        for k, v in mapping.items():
            name = name.replace(k, v)

        exp_df = eval(name)
        exp_df = normalize_factor(exp_df, y_label)
        result = self._test(exp_df, y_label)

        if result:
            return exp_df, 'pass'
        else:
            return exp_df, 'fail'


    def _test(self, exp_df: pd.DataFrame, y_label: pd.DataFrame):
        """
        Test the calculated expression against performance criteria.
        The test checks for data availability and performance metrics.
        """
        factor_mat = exp_df.values
        return_mat = y_label.values

        if self._nan_ratio(factor_mat, return_mat) < 0.9: 
            return False
        
        long_sharpe, long_max_dd, hedge_sharpe, hedge_max_dd = self._group_return(factor_mat, return_mat, y_label)
        
        if (long_sharpe >= 1.4) and (hedge_sharpe >= 2.8) and (hedge_max_dd <= 8) and (long_max_dd <= 8): 
            return True
        
        return False
    

    def _nan_ratio(self, factor_mat: np.ndarray, return_mat: np.ndarray) -> float:
        """
        Calculate the ratio of non-NaN values in the factor matrix relative to valid returns.
        """
        factor_is_nan = np.isnan(factor_mat)
        ret_is_valid = ~np.isnan(return_mat)
        target_cond = np.logical_and(factor_is_nan, ret_is_valid)
        target_count_per_row = np.sum(target_cond, axis=1)
        ret_valid_count_per_row = np.sum(ret_is_valid, axis=1)
        ratio_per_row = np.where(ret_valid_count_per_row > 0, target_count_per_row / ret_valid_count_per_row, np.nan)
        avg_ratio = np.nanmean(ratio_per_row)
        not_nan_ratio = 1 - avg_ratio
        return round(not_nan_ratio, 4)
    

    def _group_return(self, factor_mat: np.ndarray, return_mat: np.ndarray, y_label: pd.DataFrame, n_groups: int = 10):
        """
        Calculate grouped returns based on factor values.
        Instead of grouping by factor values, it takes the highest and lowest return groups as long and short positions.
        This approach is compatible with nonlinear factors.
        """
        # Data cleaning
        valid_mask = np.logical_and(~np.isnan(factor_mat), ~np.isnan(return_mat))
        valid_count = np.sum(valid_mask, axis=1)
        valid_date_mask = valid_count >= n_groups
        valid_dates = y_label.index[valid_date_mask]
        factor_mat_valid = factor_mat[valid_date_mask]
        return_mat_valid = return_mat[valid_date_mask]
        valid_mask_valid = valid_mask[valid_date_mask]
        n_valid_dates = len(valid_dates)
        ret_cols = [f'ret_{g}' for g in range(n_groups)]
        
        # Get cross-sectional rankings
        factor_rank_mat = factor_to_unique_rank(factor_mat_valid, valid_mask_valid)
        # Cross-sectional grouping
        group_labels = vectorized_rank_group(factor_rank_mat, valid_mask_valid, n_groups)
        # Calculate cross-sectional returns
        group_masks = [group_labels == g for g in range(n_groups)]
        group_rets = np.zeros((n_valid_dates, n_groups), dtype=np.float64)
        for g in range(n_groups):
            mask = np.logical_and(group_masks[g], valid_mask_valid)
            n_valid_per_date = np.sum(mask, axis=1)
            group_raw_ret = np.nansum(return_mat_valid * mask, axis=1) / np.where(n_valid_per_date > 0, n_valid_per_date, np.nan)
            group_rets[:, g] = np.where(n_valid_per_date >= 1, group_raw_ret, np.nan)
        result = pd.DataFrame(group_rets, index=valid_dates, columns=ret_cols)
        # Calculate net value trends
        cumret_cols = [f'cumret_{g}' for g in range(n_groups)]
        result[cumret_cols] = result[ret_cols].cumsum(axis=0)
        nv_cols = [f'nv_{g}' for g in range(n_groups)]
        result[nv_cols] = (1 + result[cumret_cols]).div((1 + result[cumret_cols]).iloc[0], axis=1)
        group_avg_returns = result[[f'ret_{g}' for g in range(n_groups)]].mean(axis=0)
        long_group = group_avg_returns.idxmax()
        short_group = group_avg_returns.idxmin()
        long_group_idx = int(long_group.split('_')[-1])
        result['ret_hedge'] = result[long_group] - result[short_group]
        result['ret_long'] = result[long_group]
        result['nv_long'] = result[f'nv_{long_group_idx}']
        result['cumret_hedge'] = result['ret_hedge'].cumsum()
        result['nv_hedge'] = (1 + result['cumret_hedge']) / (1 + result['cumret_hedge'].iloc[0])

        # Calculate hedged portfolio metrics
        hedge_ann_ret = ((1 + result['ret_hedge']).prod() ** (242 / result['ret_hedge'].count()) - 1)
        hedge_ann_vol = result['ret_hedge'].std() * np.sqrt(242)
        hedge_sharpe = hedge_ann_ret / hedge_ann_vol
        hedge_max_dd = self._max_dd(result['nv_hedge']) * 100
        
        # Calculate long-only portfolio metrics
        long_ann_ret = ((1 + result['ret_long']).prod() ** (242 / result['ret_long'].count()) - 1)
        long_ann_vol = result['ret_long'].std() * np.sqrt(242)
        long_sharpe = long_ann_ret / long_ann_vol
        long_max_dd = self._max_dd(result['nv_long']) * 100

        return long_sharpe, long_max_dd, hedge_sharpe, hedge_max_dd
    

    def _max_dd(self, nv: pd.Series):
        """
        Calculate maximum drawdown from net value series.
        """
        pre_max = nv.cummax()
        dd = (nv - pre_max) / pre_max
        return - np.min(dd)


    def _save(self, factor: pd.DataFrame, name: str):
        """
        Save the validated factor to a pickle file.
        The method cleans up special characters in the filename before saving.
        """
        name = name.replace('_df', '').replace('*', '×').replace('/', '÷')
        factor.to_pickle(os.path.join(self.path, f"{name}.pkl"))



class FactorTest:
    """
    FactorTest class calculates performance metrics for a given factor.
    It specifically computes the Sharpe ratio for the best-performing group of stocks.
    """

    def __init__(self):
        """
        Initialize the FactorTest with full sample return matrix and dates.
        """
        self.return_mat = FullSample.y_label.values
        self.dates = FullSample.y_label.index


    def test(self, factor: pd.DataFrame, n_groups: int = 10):
        """
        Test a factor and return its long-only Sharpe ratio.
        """
        # Data cleaning and normalization
        factor = normalize_factor(factor, FullSample.y_label)
        factor_mat = factor.values
        valid_mask = np.logical_and(~np.isnan(factor_mat), ~np.isnan(self.return_mat))
        valid_count = np.sum(valid_mask, axis=1)
        valid_date_mask = valid_count >= n_groups
        valid_dates = self.dates[valid_date_mask]
        factor_mat_valid = factor_mat[valid_date_mask]
        return_mat_valid = self.return_mat[valid_date_mask]
        valid_mask_valid = valid_mask[valid_date_mask]
        n_valid_dates = len(valid_dates)
        ret_cols = [f'ret_{g}' for g in range(n_groups)]
        
        # Get cross-sectional rankings
        factor_rank_mat = factor_to_unique_rank(factor_mat_valid, valid_mask_valid)
        # Cross-sectional grouping
        group_labels = vectorized_rank_group(factor_rank_mat, valid_mask_valid, n_groups)
        # Calculate cross-sectional returns
        group_masks = [group_labels == g for g in range(n_groups)]
        group_rets = np.zeros((n_valid_dates, n_groups), dtype=np.float64)
        for g in range(n_groups):
            mask = np.logical_and(group_masks[g], valid_mask_valid)
            n_valid_per_date = np.sum(mask, axis=1)
            group_raw_ret = np.nansum(return_mat_valid * mask, axis=1) / np.where(n_valid_per_date > 0, n_valid_per_date, np.nan)
            group_rets[:, g] = np.where(n_valid_per_date >= 1, group_raw_ret, np.nan)
        result = pd.DataFrame(group_rets, index=valid_dates, columns=ret_cols)
        # Calculate net value trends
        cumret_cols = [f'cumret_{g}' for g in range(n_groups)]
        result[cumret_cols] = result[ret_cols].cumsum(axis=0)
        nv_cols = [f'nv_{g}' for g in range(n_groups)]
        result[nv_cols] = (1 + result[cumret_cols]).div((1 + result[cumret_cols]).iloc[0], axis=1)
        group_cols = [f'ret_{g}' for g in range(n_groups)]
        group_avg_returns = result[group_cols].mean(axis=0)
        long_group = group_avg_returns.idxmax()
        long_group_idx = int(long_group.split('_')[-1])
        result['ret_long'] = result[long_group]
        result['nv_long'] = result[f'nv_{long_group_idx}']
        result = result.sort_index()

        # Calculate annualized return and volatility for the best group
        long_ann_ret = ((1 + result['ret_long']).prod() ** (242/result['ret_long'].count()) - 1)
        long_ann_vol = result['ret_long'].std() * np.sqrt(242)
        long_sharpe = long_ann_ret / long_ann_vol

        return long_sharpe


def normalize_factor(factor: pd.DataFrame, y_label: pd.DataFrame, tile: float = 0.01) -> pd.DataFrame:
    """
    Normalize factor: align with y label, winsorize cross-sectionally at 1%, standardize cross-sectionally.
    """
    factor_df_aligned = factor.reindex(
        index = y_label.index,
        columns = y_label.columns,
        fill_value=np.nan)
    arr = factor_df_aligned.values.astype(np.float64)
    pct_low = np.nanquantile(arr, tile, axis=1)
    pct_high = np.nanquantile(arr, 1-tile, axis=1)
    y_arr = y_label.values
    nan_mask = np.isnan(y_arr)
    arr[nan_mask] = np.nan
    arr_clipped = np.clip(arr,
        a_min = pct_low.reshape(-1, 1),
        a_max = pct_high.reshape(-1, 1))
    row_means = np.nanmean(arr_clipped, axis=1, keepdims=True)
    row_stds = np.nanstd(arr_clipped, axis=1, keepdims=True, ddof=0)
    arr_normalized = (arr_clipped - row_means) / row_stds
    return pd.DataFrame(
        arr_normalized,
        index = y_label.index,
        columns = y_label.columns)


def factor_to_unique_rank(mat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Convert factor values to unique ranks within each date.
    This function ensures that each factor value gets a unique rank even if there are ties.
    """
    n_dates, _ = mat.shape
    rank_mat = np.full_like(mat, np.nan, dtype=np.float64)
    for i in range(n_dates):
        valid_idx = np.where(mask[i])[0]
        if len(valid_idx) == 0:
            continue
        valid_factor = mat[i, valid_idx]
        sorted_idx = np.lexsort((valid_idx, valid_factor))
        ranks = np.arange(1, len(valid_idx) + 1)
        rank_mat[i, valid_idx[sorted_idx]] = ranks
    return rank_mat


def vectorized_rank_group(mat: np.ndarray, mask: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Assign stocks to groups based on their ranks within each date.
    """
    n_dates, n_stocks = mat.shape
    group_labels = np.full((n_dates, n_stocks), -1, dtype=np.int8)
    for i in range(n_dates):
        valid_idx = np.where(mask[i])[0]
        n_valid = len(valid_idx)
        if n_valid < n_groups:
            continue
        rank_bounds = np.linspace(0, n_valid, n_groups + 1)
        groups = np.digitize(mat[i, valid_idx], rank_bounds) - 1
        group_labels[i, valid_idx] = groups
    return group_labels