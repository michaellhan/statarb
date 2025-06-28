import pandas as pd
import numpy as np
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

class SpreadAnalysis:
    
    def __init__(self):
        self.spread = None
        self.z_scores = {}
        self.rolling_stats = {}
        
    def set_spread(self, spread: pd.Series):
        self.spread = spread
        
    def calculate_rolling_statistics(self, window_sizes: List[int]) -> dict:
        if self.spread is None:
            raise ValueError("Spread not set. Call set_spread first.")
        
        rolling_stats = {}
        
        for window in window_sizes:
            if window >= len(self.spread):
                continue
                
            rolling_mean = self.spread.rolling(window=window, min_periods=window).mean()
            rolling_std = self.spread.rolling(window=window, min_periods=window).std()
            
            rolling_stats[window] = {
                'mean': rolling_mean,
                'std': rolling_std
            }
            
        self.rolling_stats = rolling_stats
        return rolling_stats
    
    def calculate_z_scores(self, window_sizes: List[int]) -> dict:
        if not self.rolling_stats:
            self.calculate_rolling_statistics(window_sizes)
        
        z_scores = {}
        
        for window in window_sizes:
            if window not in self.rolling_stats:
                continue
                
            rolling_mean = self.rolling_stats[window]['mean']
            rolling_std = self.rolling_stats[window]['std']
            
            z_score = (self.spread - rolling_mean) / rolling_std
            
            z_scores[window] = z_score
            
        self.z_scores = z_scores
        return z_scores
    
    def get_z_score_summary(self, window_size: int) -> dict:
        if window_size not in self.z_scores:
            raise ValueError(f"Z-scores not calculated for window size {window_size}")
        
        z_score = self.z_scores[window_size].dropna()
        
        return {
            'mean': z_score.mean(),
            'std': z_score.std(),
            'min': z_score.min(),
            'max': z_score.max(),
            'count': len(z_score),
            'positive_count': (z_score > 0).sum(),
            'negative_count': (z_score < 0).sum()
        }
    
    def get_all_z_score_summaries(self) -> dict:
        summaries = {}
        
        for window_size in self.z_scores.keys():
            summaries[window_size] = self.get_z_score_summary(window_size)
            
        return summaries 