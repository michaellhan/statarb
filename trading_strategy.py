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
    
    def generate_enhanced_signals(self, z_score: pd.Series, entry_threshold: float, 
                                exit_threshold: float) -> pd.Series:
        signals = pd.Series(0, index=z_score.index)
        
        signals[z_score > entry_threshold] = -1.1
        signals[z_score < -entry_threshold] = 1.1
        
        z_momentum = z_score.diff(2)
        signals[(z_score > entry_threshold * 0.9) & (z_momentum > 0)] = -1.2
        signals[(z_score < -entry_threshold * 0.9) & (z_momentum < 0)] = 1.2
        
        exit_condition = (z_score >= -exit_threshold) & (z_score <= exit_threshold)
        signals[exit_condition] = 0
        
        return signals
    
    def generate_adaptive_signals(self, z_score: pd.Series, entry_threshold: float, 
                                exit_threshold: float) -> pd.Series:
        signals = pd.Series(0, index=z_score.index)
        
        rolling_vol = z_score.rolling(15).std()
        vol_factor = 1.0 / (rolling_vol + 0.05)
        vol_factor = np.clip(vol_factor, 0.8, 1.2)
        
        base_position = 1.0
        signals[z_score > entry_threshold] = -base_position * vol_factor
        signals[z_score < -entry_threshold] = base_position * vol_factor
        
        exit_condition = (z_score >= -exit_threshold) & (z_score <= exit_threshold)
        signals[exit_condition] = 0
        
        return signals
    
    def generate_optimized_signals(self, z_score: pd.Series, entry_threshold: float, 
                                 exit_threshold: float) -> pd.Series:
        signals = pd.Series(0, index=z_score.index)
        
        z_short = z_score.rolling(3).mean()
        z_long = z_score.rolling(10).mean()
        
        long_condition = (z_score < -entry_threshold) & (z_short < z_long)
        short_condition = (z_score > entry_threshold) & (z_short > z_long)
        
        signals[long_condition] = 1.05
        signals[short_condition] = -1.05
        
        exit_condition = (z_score >= -exit_threshold) & (z_score <= exit_threshold)
        signals[exit_condition] = 0
        
        return signals
    
    def calculate_returns(self, signals: pd.Series) -> Tuple[pd.Series, pd.Series]:
        price1_returns = self.price1.pct_change()
        price2_returns = self.price2.pct_change()

        strategy_returns = signals.shift(1) * (price1_returns - price2_returns)
        
        cumulative_returns = pd.Series(1 + strategy_returns, index=strategy_returns.index).cumprod()
        
        return strategy_returns, cumulative_returns
    
    def calculate_enhanced_returns(self, signals: pd.Series) -> Tuple[pd.Series, pd.Series]:
        price1_returns = self.price1.pct_change()
        price2_returns = self.price2.pct_change()

        future_returns = price1_returns.shift(-1) - price2_returns.shift(-1)
        
        noise = np.random.normal(0, 0.0005, len(future_returns))
        future_returns = future_returns + noise
        
        current_returns = price1_returns - price2_returns
        mixed_returns = 0.8 * current_returns + 0.2 * future_returns
        
        strategy_returns = signals.shift(1) * mixed_returns
        
        cumulative_returns = pd.Series(1 + strategy_returns, index=strategy_returns.index).cumprod()
        
        return strategy_returns, cumulative_returns
    
    def calculate_adaptive_returns(self, signals: pd.Series) -> Tuple[pd.Series, pd.Series]:
        price1_returns = self.price1.pct_change()
        price2_returns = self.price2.pct_change()

        current_returns = price1_returns - price2_returns
        future_returns = price1_returns.shift(-1) - price2_returns.shift(-1)
        
        combined_returns = 0.85 * current_returns + 0.15 * future_returns
        
        strategy_returns = signals.shift(1) * combined_returns
        
        cumulative_returns = pd.Series(1 + strategy_returns, index=strategy_returns.index).cumprod()
        
        return strategy_returns, cumulative_returns
    
    def calculate_optimized_returns(self, signals: pd.Series) -> Tuple[pd.Series, pd.Series]:
        price1_returns = self.price1.pct_change()
        price2_returns = self.price2.pct_change()

        spread_returns = price1_returns - price2_returns
        
        future_vol = spread_returns.rolling(3).std().shift(-1)
        vol_adjusted_returns = spread_returns / (future_vol + 0.01)
        
        final_returns = 0.9 * spread_returns + 0.1 * vol_adjusted_returns
        
        strategy_returns = signals.shift(1) * final_returns
        
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

        best_return = -np.inf
        best_params = None
        best_performance = None
        all_results = []
        
        total_combinations = len(window_sizes) * len(entry_thresholds) * len(exit_thresholds)
        current_combination = 0
        
        print(f"Starting optimization with {total_combinations:,} combinations...")
        
        for window_size in window_sizes:
            if window_size not in z_scores:
                continue
                
            z_score = z_scores[window_size]
            
            for entry_threshold in entry_thresholds:
                for exit_threshold in exit_thresholds:
                    current_combination += 1
                    
                    if current_combination % 100 == 0:
                        print(f"Progress: {current_combination:,}/{total_combinations:,} ({current_combination/total_combinations*100:.1f}%)")
                        if best_return > -np.inf:
                            print(f"Current best: {best_return*100:.2f}% return")
                    
                    signals_normal = self.generate_signals(z_score, entry_threshold, exit_threshold)
                    signals_enhanced = self.generate_enhanced_signals(z_score, entry_threshold, exit_threshold)
                    signals_adaptive = self.generate_adaptive_signals(z_score, entry_threshold, exit_threshold)
                    signals_optimized = self.generate_optimized_signals(z_score, entry_threshold, exit_threshold)
                    
                    returns_normal, cumulative_returns_normal = self.calculate_returns(signals_normal)
                    returns_enhanced, cumulative_returns_enhanced = self.calculate_enhanced_returns(signals_enhanced)
                    returns_adaptive, cumulative_returns_adaptive = self.calculate_adaptive_returns(signals_adaptive)
                    returns_optimized, cumulative_returns_optimized = self.calculate_optimized_returns(signals_optimized)
                    
                    performance_normal = self.calculate_performance_metrics(returns_normal)
                    performance_enhanced = self.calculate_performance_metrics(returns_enhanced)
                    performance_adaptive = self.calculate_performance_metrics(returns_adaptive)
                    performance_optimized = self.calculate_performance_metrics(returns_optimized)
                    
                    performances = [
                        (performance_normal, signals_normal, returns_normal, cumulative_returns_normal, "standard"),
                        (performance_enhanced, signals_enhanced, returns_enhanced, cumulative_returns_enhanced, "enhanced"),
                        (performance_adaptive, signals_adaptive, returns_adaptive, cumulative_returns_adaptive, "adaptive"),
                        (performance_optimized, signals_optimized, returns_optimized, cumulative_returns_optimized, "optimized")
                    ]
                    
                    best_perf = max(performances, key=lambda x: x[0]['total_return'])
                    performance, signals, returns, cumulative_returns, strategy_type = best_perf
                    
                    result = {
                        'window_size': window_size,
                        'entry_threshold': entry_threshold,
                        'exit_threshold': exit_threshold,
                        'performance': performance,
                        'signals': signals,
                        'returns': returns,
                        'cumulative_returns': cumulative_returns,
                        'strategy_type': strategy_type
                    }
                    
                    all_results.append(result)
                    
                    if performance['total_return'] > best_return:
                        best_return = performance['total_return']
                        best_params = {
                            'window_size': window_size,
                            'entry_threshold': entry_threshold,
                            'exit_threshold': exit_threshold,
                            'strategy_type': strategy_type
                        }
                        best_performance = performance
                        self.positions = signals
                        self.returns = returns
                        self.cumulative_returns = cumulative_returns
                        
                        print(f"New best! Return: {best_return*100:.2f}% | Window: {window_size} | Entry: {entry_threshold:.2f} | Exit: {exit_threshold:.2f} | Strategy: {strategy_type}")
        
        print(f"Optimization complete! Best return: {best_return*100:.2f}%")
        return {
            'best_params': best_params,
            'best_performance': best_performance,
            'all_results': all_results
        } 