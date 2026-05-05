import pandas as pd
import numpy as np
import os

from Config import RootConfig



def merge_data():
    """
    Merges all raw fetched data sources into a single comprehensive DataFrame.
    The final merged dataset is saved as 'tot_data.pkl'.
    """
    adjust_data = pd.read_pickle(os.path.join(RootConfig.DATA_ROOT, 'adjust_data.pkl'))
    stock_basic_data = pd.read_pickle(os.path.join(RootConfig.DATA_ROOT, 'stock_basic_data.pkl'))
    st_data = pd.read_pickle(os.path.join(RootConfig.DATA_ROOT, 'st_data.pkl'))
    daily_basic_data = pd.read_pickle(os.path.join(RootConfig.DATA_ROOT, 'daily_basic_data.pkl'))
    limit_price_data = pd.read_pickle(os.path.join(RootConfig.DATA_ROOT, 'limit_price_data.pkl'))
    suspend_data = pd.read_pickle(os.path.join(RootConfig.DATA_ROOT, 'suspend_data.pkl'))
    daily_data = pd.read_pickle(os.path.join(RootConfig.DATA_ROOT, 'daily_data.pkl'))
    daily_data = daily_data.rename(columns={'open': 'origin_open'})
    zz1000_weight = pd.read_pickle(os.path.join(RootConfig.DATA_ROOT, 'zz1000_weight.pkl'))
    zz1000_weight = zz1000_weight.rename(columns={'con_code': 'ts_code'})
    zz1000_weight['month'] = zz1000_weight['trade_date'].astype('str').str[:6]
    zz1000_weight = zz1000_weight.drop(columns=['weight', 'trade_date'])
    zz1000_weight = zz1000_weight.drop_duplicates(subset=['ts_code', 'month'])

    tot_data = pd.merge(adjust_data, stock_basic_data, on=['ts_code'], how='left')
    tot_data = pd.merge(tot_data, st_data[['ts_code', 'trade_date', 'type']], on=['ts_code', 'trade_date'], how='left')
    tot_data = tot_data.rename(columns={'type': 'st_status'})
    tot_data = pd.merge(tot_data, daily_basic_data[['ts_code', 'trade_date', 'turnover_rate_f','circ_mv']], on=['ts_code', 'trade_date'], how='left')
    tot_data = pd.merge(tot_data, limit_price_data, on=['ts_code', 'trade_date'], how='left')
    tot_data = pd.merge(tot_data, suspend_data[['ts_code', 'trade_date', 'suspend_type']], on=['ts_code', 'trade_date'], how='left')
    tot_data = pd.merge(tot_data, daily_data[['ts_code','trade_date','origin_open']], on=['ts_code','trade_date'], how='left')
    tot_data['month'] = tot_data['trade_date'].astype('str').str[:6]
    tot_data = pd.merge(tot_data, zz1000_weight, on=['ts_code', 'month'], how='left')
    tot_data = tot_data.drop(columns=['month'])
    tot_data.to_pickle(os.path.join(RootConfig.DATA_ROOT, 'tot_data.pkl'))


def process_data():
    """
    Performs extensive data cleaning, filtering, and label engineering.

    Key steps include:
    1.  Filtering: Removes stocks from unwanted markets BSE,
        suspended stocks, ST stocks, and stocks listed for less than one year.
    2.  Price Cleaning: Filters out stocks that hit opening price limits.
    3.  Label Generation: Creates the prediction target 'y_label', defined as 
        the return from the open price of t+1 to t+2: (Open_t+2 - Open_t+1) / Open_t+1.
    4.  Saving: Saves the cleaned price data and the specific label DataFrame 
        for the CSI 1000 universe.
    """
    # tot_data = pd.read_pickle(os.path.join(RootConfig.DATA_ROOT, 'tot_data.pkl'))
    tot_data = pd.read_pickle(os.path.join(r'F:\derek\quant\strat1\data_base', 'tot_data.pkl'))
    tot_data = tot_data.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    tot_data = tot_data.drop(columns=['change', 'pct_chg', 'pre_close'])

    # Remove not target market
    tot_data['not_target'] = ((tot_data['exchange'] == 'BSE') | (tot_data['market'] == '北交所') | 
                            (tot_data['market'] == '科创板') | (tot_data['ts_code'].str[-2:] == 'BJ'))
    tot_data = tot_data[tot_data['not_target'] == False]
    tot_data = tot_data.drop(columns=['not_target','market','exchange'])

    # Remove suspended stock
    tot_data = tot_data[tot_data['suspend_type'].isna()]
    tot_data = tot_data.drop(columns=['suspend_type'])

    # Convert price data to wide DataFrame and save
    price = tot_data[['ts_code','trade_date','open','high','low','close',
                  'vol','amount','turnover_rate_f','circ_mv']]
    price.pivot(index='trade_date',columns='ts_code',values='open').sort_index(axis=0,ascending=True).sort_index(axis=1,ascending=True).to_pickle(os.path.join(RootConfig.DATA_ROOT, 'open.pkl'))
    price.pivot(index='trade_date',columns='ts_code',values='close').sort_index(axis=0,ascending=True).sort_index(axis=1,ascending=True).to_pickle(os.path.join(RootConfig.DATA_ROOT, 'close.pkl'))
    price.pivot(index='trade_date',columns='ts_code',values='high').sort_index(axis=0,ascending=True).sort_index(axis=1,ascending=True).to_pickle(os.path.join(RootConfig.DATA_ROOT, 'high.pkl'))
    price.pivot(index='trade_date',columns='ts_code',values='low').sort_index(axis=0,ascending=True).sort_index(axis=1,ascending=True).to_pickle(os.path.join(RootConfig.DATA_ROOT, 'low.pkl'))
    price.pivot(index='trade_date',columns='ts_code',values='vol').sort_index(axis=0,ascending=True).sort_index(axis=1,ascending=True).to_pickle(os.path.join(RootConfig.DATA_ROOT, 'volume.pkl'))
    price.pivot(index='trade_date',columns='ts_code',values='amount').sort_index(axis=0,ascending=True).sort_index(axis=1,ascending=True).to_pickle(os.path.join(RootConfig.DATA_ROOT, 'amount.pkl'))
    price.pivot(index='trade_date',columns='ts_code',values='turnover_rate_f').sort_index(axis=0,ascending=True).sort_index(axis=1,ascending=True).to_pickle(os.path.join(RootConfig.DATA_ROOT, 'turn.pkl'))
    price.pivot(index='trade_date',columns='ts_code',values='circ_mv').sort_index(axis=0,ascending=True).sort_index(axis=1,ascending=True).to_pickle(os.path.join(RootConfig.DATA_ROOT, 'marketcap.pkl'))

    # Remove stocks reached the opening price limit
    tot_data['open_limit'] = np.where((tot_data['origin_open']>=tot_data['up_limit'])|
            (tot_data['origin_open']<=tot_data['down_limit']), True, False)
    tot_data = tot_data[tot_data['open_limit']==False]
    tot_data = tot_data.drop(columns=['open_limit', 'up_limit','down_limit', 'origin_open'])

    # y_label = (Open_t+2 - Open_t+1) / Open_t+1
    tot_data['next_open'] = tot_data.groupby('ts_code')['open'].shift(-1)
    tot_data['next_next_open'] = tot_data.groupby('ts_code')['open'].shift(-2)
    tot_data['y_label'] = tot_data['next_next_open'] / tot_data['next_open'] - 1
    tot_data = tot_data.drop(columns=['next_open', 'next_next_open'])

    # Remove st stocks
    tot_data = tot_data[tot_data['st_status'].isna()]
    tot_data = tot_data.drop(columns='st_status')

    # Remove stocks listed for less than one year
    tot_data["date1"] = pd.to_datetime(tot_data["trade_date"])
    tot_data["date2"] = pd.to_datetime(tot_data["list_date"])
    tot_data["less_than_1_y"] = (tot_data["date1"] - tot_data["date2"]) < pd.Timedelta(days=365)
    tot_data = tot_data[tot_data['less_than_1_y'] == False]
    tot_data = tot_data.drop(columns=['date1', 'date2', 'less_than_1_y'])

    # Save CSI 1000 universe stocks' label
    label = tot_data[tot_data['index_code']=='000852.SH']
    (label.pivot(index='trade_date', columns='ts_code', values='y_label')
     .sort_index(axis=0, ascending=True).sort_index(axis=1, ascending=True)
     .to_pickle(os.path.join(RootConfig.DATA_ROOT, 'zz1000_label.pkl')))



if __name__ == '__main__':
    merge_data()
    process_data()