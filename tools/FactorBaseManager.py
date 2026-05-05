import pandas as pd
import numpy as np
from typing import List
import os
import glob
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")

from tools.FactorTest import FactorTest


"""
Factor Base Management:

Main purpose: Remove highly correlated factors to achieve dimensionality reduction

Processing methods:
1. Obtain absolute correlation matrix of factor base
2. Based on correlation matrix, get pairwise correlations between factors
3. Process factors from high to low correlation
4. For factor pairs with correlation higher than 95%, retain the factor with shorter expression. Because correlation higher than 95% means they are almost identical factors, shorter expressions have lower complexity and overfitting, higher interpretability and computational performance
5. For factor pairs with correlation between 70%~95%, compare the long-sharpe ratio of factors and retain the higher one
"""


class FactorBaseManager:
    """
    FactorBaseManager handles the management of factor libraries by removing highly correlated factors
    to achieve dimensionality reduction. It implements a systematic approach to identify and eliminate
    redundant factors based on correlation thresholds and performance metrics.
    """
    
    def __init__(self, factor_base_path: str, max_files: int):
        """
        Initialize the FactorBaseManager with path to factor library and maximum number of files to process.
        """
        self.factor_data_base = factor_base_path
        self._read_factors(factor_base_path, max_files)
        self.factor_test = FactorTest()


    def remove_high_coor_factors(self):
        """
        Execute the process to remove highly correlated factors from the factor base.
        This method orchestrates the entire workflow of correlation calculation and factor removal.
        """
        self._cal_corr_matrix(self.dates)
        print("Starting to remove highly correlated factors.")
        self.high_corr = self._get_high_correlation_pairs(0.7)
        self._remove_high_corr_method1()
        self._remove_high_corr_method2()
        print("Factor base management completed.")

    
    def _read_factors(self, path: str, max_files: int):
        """
        Detect all factor files in the path, read them one by one, and check factor format matching.
        """
        # Detect all factor files
        file_paths = glob.glob(os.path.join(path, "*.pkl"))
        file_paths = sorted(file_paths, key = lambda x: os.path.basename(x))

        # Limit the number of files to process
        if len(file_paths) >= max_files:
            file_paths = file_paths[:max_files]
        print(f"发现 {len(file_paths)} 个因子文件，开始处理...")

        factor_values = []
        factor_names = []

        # Read the first factor and use its index and column names as reference
        value, base_dates, base_tickers, name = self._read_wide_array(file_paths[0])
        factor_values.append(value)
        factor_names.append(name)

        # Multi-threaded reading of remaining files, checking format consistency
        with ThreadPoolExecutor(max_workers = 8) as executor:
            future_to_path = {executor.submit(self._read_wide_array, path): path for path in file_paths[1:]}
            for future in as_completed(future_to_path):
                value, dates, tickers, name = future.result()
                if dates != base_dates:
                    print(f"File {name} structure inconsistent, dates don't match")
                    continue
                elif tickers != base_tickers:
                    print(f"File {name} structure inconsistent, stocks don't match")
                    continue
                else:
                    factor_values.append(value)
                    factor_names.append(name)

        self.factor_names = factor_names
        self.factors = self._reshape_values(factor_values, factor_names, base_dates, base_tickers)
        self.dates = base_dates


    def _reshape_values(self, values, names, dates, tickers) -> pd.DataFrame: 
        """
        Efficiently and memory-savingly convert factor list to factor long table.
        """
        T, N, M = len(dates), len(tickers), len(names)
        factor_tensor = np.stack(values, axis=-1) # (T,N,M)
        del values
        reshaped_data = factor_tensor.reshape(T * N, M)
        del factor_tensor

        datetimes = np.repeat(dates, N) 
        tickers = np.tile(tickers, T)
        df = pd.DataFrame(
            data = reshaped_data,
            columns = names)
        df.insert(0, "ts_code", tickers)
        df.insert(0, "trade_date", datetimes) 

        return df.set_index(['trade_date','ts_code'])


    def _cal_corr_matrix(self, dates):
        """
        Calculate daily factor inter-absolute correlation matrix, then take average.
        """
        print("Starting correlation calculation.")    
        arrays = []
        factor_df = self.factors.dropna(axis=1,how='all').dropna(axis=0,how='all')

        for date in dates:
            try:
                df_date = factor_df.xs(date, level='trade_date')
            except:
                continue
            
            if len(df_date) < 2:
                print(f"Warning: Insufficient data for date {date}, skipping")
                continue

            df_date = df_date.fillna(0)
            values_array = df_date.values.astype(float)
            corr_matrix_np = np.corrcoef(values_array, rowvar=False)
            arrays.append(corr_matrix_np)

        stacked = np.stack(arrays)
        mean_corr = np.abs(np.nanmean(stacked, axis=0))

        self.corr_matrix = pd.DataFrame(mean_corr, 
                                   index=self.factor_names, 
                                   columns=self.factor_names)
        
    
    def _get_high_correlation_pairs(self, threshold) -> pd.DataFrame:
        """
        Get factor pairs with correlation higher than threshold.
        """
        
        high_corr_pairs = []
        upper_triangle = self.corr_matrix.where(np.triu(np.ones(self.corr_matrix.shape), k=1).astype(bool))
        for col in upper_triangle.columns:
            for idx in upper_triangle.index:
                corr_value = upper_triangle.loc[idx, col]
                if not pd.isna(corr_value) and corr_value >= threshold:
                    high_corr_pairs.append({
                        'Factor1': col,
                        'Factor2': idx,
                        'Abs Corr': corr_value})
        result_df = pd.DataFrame(high_corr_pairs)

        if not result_df.empty:
            result_df = result_df.sort_values(by='Abs Corr',ascending=False).reset_index(drop=True)

        return result_df
    
    
    def _read_wide_array(self, file_path):
        """
        Read factor wide table, return: values, index, column names, file name.
        """
        df_wide = pd.read_pickle(file_path)
        df_wide = df_wide[(df_wide.index >= '20170101')&(df_wide.index < '20250101')]
        datetimes = df_wide.index.to_list()
        tickers = df_wide.columns.to_list()
        factor_array = df_wide.values
        factor_name = os.path.splitext(os.path.basename(file_path))[0]
        return factor_array, datetimes, tickers, factor_name
    

    def _remove_high_corr_method1(self, corr_bar = 0.95):
        """
        For factor pairs with correlation higher than 95%, retain the factor with shorter expression.
        Shorter expressions have lower complexity and overfitting risk, 
        higher interpretability and computational performance.
        """
        if self.high_corr.empty:
            return
        
        high_corr = self.high_corr[self.high_corr['Abs Corr'] >= corr_bar]

        if (high_corr.empty): 
            return
        
        remove_factors = []
        for i in high_corr.index:
            factor_name1 = high_corr.loc[i,'Factor1']
            factor_name2 = high_corr.loc[i,'Factor2']
            if (factor_name1 in remove_factors) | (factor_name2 in remove_factors):
                continue
            if len(factor_name1) > len(factor_name2):
                remove_factors.append(factor_name1)
            else:
                remove_factors.append(factor_name2)

        self._delete_factors(remove_factors)

    
    def _remove_high_corr_method2(self, corr_bar = 0.7):
        """
        For factor pairs with correlation between 70%~95%, 
        compare the long-sharpe ratio of factors and retain the higher one.
        """
        if self.high_corr.empty:
            return
        high_corr = self.high_corr[self.high_corr['Abs Corr'] >= corr_bar]

        if high_corr.empty: 
            return
        
        remove_factors = []
        for i in high_corr.index:
            factor_name1 = high_corr.loc[i,'Factor1']
            factor_name2 = high_corr.loc[i,'Factor2']
            if (factor_name1 in remove_factors) | (factor_name2 in remove_factors):
                continue
            remove_factors.append(self._compare_factors(factor_name1, factor_name2))
        
        self._delete_factors(remove_factors)


    def _delete_factors(self, drop_factor_names: List):
        """
        Delete factors from the correlation matrix and file system.
        """
        for name in drop_factor_names:
            self.high_corr = self.high_corr.drop(self.high_corr[(self.high_corr['Factor1'] == name) | (self.high_corr['Factor2'] == name)].index)

        files_to_delete = []
        for i in drop_factor_names:
            path = os.path.join(self.factor_data_base, f"{i}.pkl")
            files_to_delete.append(path)

        def remove_file(path):
            try:
                os.remove(path)
                return True, path
            except Exception as e:
                return False, f"{path} (错误: {str(e)})"

        with ThreadPoolExecutor(max_workers = 16) as executor:
            future_to_path = {executor.submit(remove_file, f): f for f in files_to_delete}
            for future in as_completed(future_to_path):
                success, result = future.result()
                if not success:
                    print(f"删除文件失败: {result}")

        self.factors = self.factors.drop(columns=drop_factor_names)
        for name in drop_factor_names:
            if name in self.factor_names:
                self.factor_names.remove(name) 


    def _compare_factors(self, factor_name1, factor_name2):
        """
        Compare two factors based on their Sharpe ratios and return the name of the factor to remove.
        """
        factor_df = self.factors.reset_index().pivot(index = 'trade_date', columns = 'ts_code', values = factor_name1)
        sharpe1 = self.factor_test.test(factor_df)
        factor_df = self.factors.reset_index().pivot(index = 'trade_date', columns = 'ts_code', values = factor_name2)
        sharpe2 = self.factor_test.test(factor_df)

        if sharpe1 > sharpe2:
            remove_factor_name = factor_name2
        else:
            remove_factor_name = factor_name1

        return remove_factor_name