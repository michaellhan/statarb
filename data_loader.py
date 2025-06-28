import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    
    def __init__(self, start_date: str = "2020-01-01", end_date: str = "2024-12-31"):
        self.start_date = start_date
        self.end_date = end_date
        
    def download_data(self, ticker1: str, ticker2: str) -> Tuple[pd.Series, pd.Series]:
        stock1 = yf.download(ticker1, start=self.start_date, end=self.end_date, progress=False)
        stock2 = yf.download(ticker2, start=self.start_date, end=self.end_date, progress=False)
        
        if stock1 is None or stock1.empty:
            raise ValueError(f"Failed to download data for {ticker1}")
        if stock2 is None or stock2.empty:
            raise ValueError(f"Failed to download data for {ticker2}")
        
        def get_price_column(df, ticker):
            if 'Adj Close' in df.columns:
                return df['Adj Close']
            elif 'Close' in df.columns:
                return df['Close']
            else:
                raise ValueError(f"No suitable price column found for {ticker}. Available columns: {list(df.columns)}")
        
        if isinstance(stock1, pd.DataFrame) and isinstance(stock1.columns, pd.MultiIndex):
            if ('Adj Close', ticker1) in stock1.columns:
                stock1_adj_close = stock1[('Adj Close', ticker1)].dropna()
            elif ('Close', ticker1) in stock1.columns:
                stock1_adj_close = stock1[('Close', ticker1)].dropna()
            else:
                raise ValueError(f"No suitable price column found for {ticker1}. Available columns: {list(stock1.columns)}")
        else:
            stock1_adj_close = get_price_column(stock1, ticker1).dropna()
        if isinstance(stock2, pd.DataFrame) and isinstance(stock2.columns, pd.MultiIndex):
            if ('Adj Close', ticker2) in stock2.columns:
                stock2_adj_close = stock2[('Adj Close', ticker2)].dropna()
            elif ('Close', ticker2) in stock2.columns:
                stock2_adj_close = stock2[('Close', ticker2)].dropna()
            else:
                raise ValueError(f"No suitable price column found for {ticker2}. Available columns: {list(stock2.columns)}")
        else:
            stock2_adj_close = get_price_column(stock2, ticker2).dropna()
        if isinstance(stock1_adj_close.index, pd.MultiIndex):
            stock1_adj_close.index = stock1_adj_close.index.get_level_values(0)
        if isinstance(stock2_adj_close.index, pd.MultiIndex):
            stock2_adj_close.index = stock2_adj_close.index.get_level_values(0)
        
        common_dates = stock1_adj_close.index.intersection(stock2_adj_close.index)
        stock1_aligned = stock1_adj_close.loc[common_dates]
        stock2_aligned = stock2_adj_close.loc[common_dates]
        
        return stock1_aligned, stock2_aligned
    
    def create_price_dataframe(self, stock1_data: pd.Series, stock2_data: pd.Series, 
                              ticker1: str, ticker2: str) -> pd.DataFrame:
        df = pd.DataFrame({
            ticker1: stock1_data,
            ticker2: stock2_data
        })
        return df.dropna() 