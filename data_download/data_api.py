import pandas as pd
from tqdm import tqdm
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from Config import RootConfig, TokenConfig
import tushare as ts

ts.set_token(TokenConfig.TS_TOKEN)
pro = ts.pro_api()
warnings.filterwarnings("ignore")



class StockDataFetcher:
    """
    Getting stock data from Tushare
    """
    
    def __init__(self):
        self.data_root = RootConfig.DATA_ROOT
        

    def fetch_stock_basic(self):
        """
        Getting stock list
        """
        stock_basic_data = pro.stock_basic(
            fields='ts_code, name, area, industry, market, exchange, list_status, list_date, delist_date'
        )
        stock_basic_data = stock_basic_data.sort_values('ts_code').reset_index(drop=True)
        output_path = os.path.join(self.data_root, 'stock_basic_data.pkl')
        stock_basic_data.to_pickle(output_path)
        print("save stock basic data")
        return stock_basic_data
    

    def fetch_trade_calendar(self):
        """
        Getting trade calendar
        """
        trade_dates = pro.trade_cal(
            start_date='20160101', 
            end_date='20251231', 
            fields='cal_date,is_open'
        )
        trade_dates = trade_dates[trade_dates['is_open']==1].sort_values('cal_date').reset_index(drop=True)
        trade_dates = trade_dates.rename(columns={'cal_date':'trade_date'}).drop(columns=['is_open'])
        output_path = os.path.join(self.data_root, 'trade_dates.pkl')
        trade_dates.to_pickle(output_path)
        print("save trade dates")
        return trade_dates
    

    def fetch_all_daily_data(self, dates, max_workers=1):
        """
        Getting all daily data
        """
        def fetch_daily_data_by_date(date):
            """
            Getting data by date
            """
            # historical data
            daily = pro.daily(start_date=date, end_date=date)
            # daily indicators
            daily_basic = pro.daily_basic(start_date=date, end_date=date)
            # ST list
            st = pro.stock_st(trade_date=date)
            # limit price
            limit_price = pro.stk_limit(trade_date=date)
            # suspended
            suspend = pro.suspend_d(suspend_type='S', trade_date=date)

            return {
                'date': date,
                'daily': daily,
                'daily_basic': daily_basic,
                "st": st,
                'limit_price': limit_price,
                'suspend': suspend,
            }
        
        def save_data_list(data_list: list, name: str):
            """
            Save data to pickle
            """
            if not data_list: 
                print(f"Warning: {name} is empty!")
                return
            data = pd.concat(data_list, axis=0, ignore_index=True)
            data = data.sort_values(['ts_code','trade_date']).reset_index(drop=True)
            output_path = os.path.join(self.data_root, f'{name}.pkl')
            data.to_pickle(output_path)
            print(f"save {name}")
        
        daily_data_list = []
        daily_basic_list = []
        st_data_list = []
        limit_price_data_list = []
        suspend_data_list = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_date = {
                executor.submit(fetch_daily_data_by_date, date): date 
                for date in dates
            }
            for future in tqdm(as_completed(future_to_date), total=len(dates)):
                result = future.result()
                daily_data_list.append(result['daily'])
                daily_basic_list.append(result['daily_basic'])
                st_data_list.append(result['st'])
                limit_price_data_list.append(result['limit_price'])
                suspend_data_list.append(result['suspend'])

        save_data_list(daily_data_list, 'daily_data')
        save_data_list(daily_basic_list, 'daily_basic_data')
        save_data_list(st_data_list, 'st_data')
        save_data_list(limit_price_data_list, 'limit_price_data')
        save_data_list(suspend_data_list, 'suspend_data')

    
    def fetch_adjusted_price(self, tickers, max_workers=1):
        """
        Getting adjusted price data
        """
        def fetch_one_ticker(ticker):
            return ts.pro_bar(
                ts_code=ticker, 
                adj='qfq', 
                start_date='20160101', 
                end_date='20251231'
            )

        adjust_data_list = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(fetch_one_ticker, ticker): ticker 
                for ticker in tickers
            }
            for future in tqdm(as_completed(future_to_ticker), total=len(tickers)):
                result = future.result()
                adjust_data_list.append(result)
        
        adjust_data = pd.concat(adjust_data_list, axis=0, ignore_index=True)
        output_path = os.path.join(self.data_root, 'adjust_data.pkl')
        adjust_data.to_pickle(output_path)
        print(f"save adjust data")
    

    def fetch_index_weight(self, name, index_code='000852.SH'):
        """
        Getting index weight
        """
        index_list = []
        years = ['2016','2017','2018','2019','2020','2021','2022','2023','2024','2025']
        months = ['01','02','03','04','05','06','07','08','09','10','11','12']

        for year in tqdm(years):
            for month in months:
                start = year + month + '01'
                end = year + month + '31'
                temp = pro.index_weight(
                    index_code=index_code,
                    start_date=start,
                    end_date=end
                )
                if not temp.empty: 
                    index_list.append(temp)
            time.sleep(27)

        index_weight = pd.concat(index_list, axis=0, ignore_index=True)
        index_weight.drop_duplicates(
            subset=['con_code','trade_date'], 
            keep='last', 
            inplace=True
        )
        output_path = os.path.join(self.data_root, name + '_weight')
        index_weight.to_pickle(output_path)
        print("save index weight")

        index_nv = pro.index_daily(ts_code=index_code,start_date='20160101',end_date='20251231')
        index_nv = index_nv.sort_values('trade_date').reset_index(drop=True)
        index_nv.to_pickle(os.path.join(RootConfig.DATA_ROOT, f'{name}.pkl'))
        print("save index net value")

    
    def run_full_fetch(self):
        stock_basic_data = self.fetch_stock_basic()

        trade_dates = self.fetch_trade_calendar()

        dates = trade_dates['trade_date'].values.tolist()
        self.fetch_all_daily_data(dates)

        tickers = stock_basic_data['ts_code'].unique().tolist()
        tickers.sort()
        self.fetch_adjusted_price(tickers)
        
        self.fetch_index_weight('zz1000', '000852.SH')
        


if __name__ == "__main__":
    fetcher = StockDataFetcher()
    fetcher.run_full_fetch()
