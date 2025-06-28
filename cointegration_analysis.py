import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

class CointegrationAnalysis:
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.is_cointegrated = False
        self.hedge_ratio = None
        self.cointegration_pvalue = None
        self.cointegration_statistic = None
        
    def test_cointegration(self, price1: pd.Series, price2: pd.Series) -> bool:
        statistic, pvalue, _ = coint(price1, price2)
        
        self.cointegration_statistic = statistic
        self.cointegration_pvalue = pvalue
        self.is_cointegrated = bool(pvalue < self.significance_level)
        
        return self.is_cointegrated
    
    def calculate_hedge_ratio(self, price1: pd.Series, price2: pd.Series) -> float:
        X = np.column_stack([np.ones(len(price2)), price2])
        model = OLS(price1, X).fit()
        self.hedge_ratio = model.params[1]
        return self.hedge_ratio
    
    def calculate_spread(self, price1: pd.Series, price2: pd.Series) -> pd.Series:
        if self.hedge_ratio is None:
            raise ValueError("Hedge ratio not calculated. Run calculate_hedge_ratio first.")
        
        spread = price1 - self.hedge_ratio * price2
        return spread
    
    def get_cointegration_summary(self) -> dict:
        return {
            'is_cointegrated': self.is_cointegrated,
            'hedge_ratio': self.hedge_ratio,
            'p_value': self.cointegration_pvalue,
            'statistic': self.cointegration_statistic,
            'significance_level': self.significance_level
        } 