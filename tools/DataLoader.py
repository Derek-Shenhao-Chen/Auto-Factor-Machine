import pandas as pd
import os
from typing import Tuple

from Config import RootConfig
from tools.FunctionSet import *



def read_file(root: str, name: str, year_range: Tuple[str, str]) -> pd.DataFrame:
    """
    Read and filter data from a pickle file based on specified date range.
    """
    df = pd.read_pickle(os.path.join(root, f'{name}.pkl'))
    df = df[(df.index >= year_range[0]) & (df.index < year_range[1])]
    df = df.sort_index(axis=0, ascending=True).sort_index(axis=1, ascending=True)
    return df


def process_y_label(y_label: pd.DataFrame) -> pd.DataFrame:
    """
    Process the y-label data by performing cross-sectional de-meaning.
    """
    arr = y_label.values.astype(np.float64)
    row_means = np.nanmean(arr, axis=1, keepdims=True)
    arr_normalized = arr - row_means
    return pd.DataFrame(
        arr_normalized,
        index = y_label.index,
        columns = y_label.columns)


class FullSample:
    """
    Class containing full sample financial data from 2017 to 2025.
    This class loads and processes financial datasets including prices,
    volumes, returns.
    """
    close_df = read_file(RootConfig.DATA_ROOT, 'close', ('20170101', '20250101'))
    open_df = read_file(RootConfig.DATA_ROOT, 'open', ('20170101', '20250101'))
    high_df = read_file(RootConfig.DATA_ROOT, 'high', ('20170101', '20250101'))
    low_df = read_file(RootConfig.DATA_ROOT, 'low', ('20170101', '20250101'))
    volume_df = read_file(RootConfig.DATA_ROOT, 'volume', ('20170101', '20250101'))
    amount_df = read_file(RootConfig.DATA_ROOT, 'amount', ('20170101', '20250101'))
    turn_df = read_file(RootConfig.DATA_ROOT, 'turn', ('20170101', '20250101'))
    marketcap_df = read_file(RootConfig.DATA_ROOT, 'marketcap', ('20170101', '20250101'))
    ret_df = pct_change_one(close_df)
    market_df = cs_mean(ret_df)
    excess_df = ret_df - market_df

    y_label = read_file(RootConfig.DATA_ROOT, 'zz1000_label', ('20170101', '20250101'))
    y_label = process_y_label(y_label)



class SmallSample:
    """
    Class containing small sample financial data from 2022 to 2025.
    """
    close_df = read_file(RootConfig.DATA_ROOT, 'close', ('20220101', '20250101'))
    open_df = read_file(RootConfig.DATA_ROOT, 'open', ('20220101', '20250101'))
    high_df = read_file(RootConfig.DATA_ROOT, 'high', ('20220101', '20250101'))
    low_df = read_file(RootConfig.DATA_ROOT, 'low', ('20220101', '20250101'))
    volume_df = read_file(RootConfig.DATA_ROOT, 'volume', ('20220101', '20250101'))
    amount_df = read_file(RootConfig.DATA_ROOT, 'amount', ('20220101', '20250101'))
    turn_df = read_file(RootConfig.DATA_ROOT, 'turn', ('20220101', '20250101'))
    marketcap_df = read_file(RootConfig.DATA_ROOT, 'marketcap', ('20220101', '20250101'))
    ret_df = pct_change_one(close_df)
    market_df = cs_mean(ret_df)
    excess_df = ret_df - market_df

    y_label = read_file(RootConfig.DATA_ROOT, 'zz1000_label', ('20220101', '20250101'))
    y_label = process_y_label(y_label)
