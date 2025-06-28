import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

class TradingStrategy:
    
    def __init__(self, price1: pd.Series, price2: pd.Series, spread: pd.Series, 
                 ticker1: str, ticker2: str):
        self.price1 = price1
        self.price2 = price2
        self.spread = spread
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.positions = None
        self.returns = None
        self.cumulative_returns = None
        
    def generate_signals(self, z_score: pd.Series, entry_threshold: float, 
                        exit_threshold: float) -> pd.Series:
        signals = pd.Series(0, index=z_score.index)
        
        signals[z_score > entry_threshold] = -1
        signals[z_score < -entry_threshold] = 1
        
        exit_condition = (z_score >= -exit_threshold) & (z_score <= exit_threshold)
        signals[exit_condition] = 0
        
        return signals
    
    def calculate_returns(self, signals: pd.Series) -> Tuple[pd.Series, pd.Series]:
        price1_returns = self.price1.pct_change()
        price2_returns = self.price2.pct_change()

        strategy_returns = signals.shift(1) * (price1_returns - price2_returns)
        
        cumulative_returns = pd.Series(1 + strategy_returns, index=strategy_returns.index).cumprod()
        
        return strategy_returns, cumulative_returns
    
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        returns_clean = returns.dropna()
        
        if len(returns_clean) == 0:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'num_trades': 0,
                'volatility': 0
            }
        
        total_return = (1 + returns_clean).prod() - 1
        volatility = returns_clean.std() * np.sqrt(252)
        sharpe_ratio = (returns_clean.mean() * 252) / volatility if volatility > 0 else 0
        
        returns_array = np.array(returns_clean.values)
        cumulative_returns_array = np.cumprod(1 + returns_array)
        running_max_array = np.maximum.accumulate(cumulative_returns_array)
        drawdown_array = (cumulative_returns_array - running_max_array) / running_max_array
        max_drawdown = np.min(drawdown_array)
        
        winning_trades = (returns_clean > 0).sum()
        total_trades = len(returns_clean)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': total_trades,
            'volatility': volatility
        }
    
    def optimize_parameters(self, z_scores: Dict[int, pd.Series], 
                          window_sizes: List[int],
                          entry_thresholds: List[float],
                          exit_thresholds: List[float]) -> Dict:

        best_sharpe = -np.inf
        best_params = None
        best_performance = None
        all_results = []
        
        for window_size in window_sizes:
            if window_size not in z_scores:
                continue
                
            z_score = z_scores[window_size]
            
            for entry_threshold in entry_thresholds:
                for exit_threshold in exit_thresholds:
                    signals = self.generate_signals(z_score, entry_threshold, exit_threshold)
                    returns, cumulative_returns = self.calculate_returns(signals)
                    performance = self.calculate_performance_metrics(returns)
                    
                    result = {
                        'window_size': window_size,
                        'entry_threshold': entry_threshold,
                        'exit_threshold': exit_threshold,
                        'performance': performance,
                        'signals': signals,
                        'returns': returns,
                        'cumulative_returns': cumulative_returns
                    }
                    
                    all_results.append(result)
                    
                    if performance['sharpe_ratio'] > best_sharpe:
                        best_sharpe = performance['sharpe_ratio']
                        best_params = {
                            'window_size': window_size,
                            'entry_threshold': entry_threshold,
                            'exit_threshold': exit_threshold
                        }
                        best_performance = performance
                        self.positions = signals
                        self.returns = returns
                        self.cumulative_returns = cumulative_returns
        
        return {
            'best_params': best_params,
            'best_performance': best_performance,
            'all_results': all_results
        } 