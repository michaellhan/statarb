import pandas as pd
import numpy as np
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from cointegration_analysis import CointegrationAnalysis
from spread_analysis import SpreadAnalysis
from trading_strategy import TradingStrategy

def main():
    TICKER1 = "KO"
    TICKER2 = "PEP"
    START_DATE = "2022-01-01"
    END_DATE = "2024-12-31"
    
    WINDOW_SIZES = list(range(3, 11, 1))
    ENTRY_THRESHOLDS = np.arange(0.1, 2.1, 0.1)
    EXIT_THRESHOLDS = np.arange(0.0, 1.1, 0.1)
    
    data_loader = DataLoader(START_DATE, END_DATE)
    stock1_data, stock2_data = data_loader.download_data(TICKER1, TICKER2)
    price_df = data_loader.create_price_dataframe(stock1_data, stock2_data, TICKER1, TICKER2)
    
    coint_analyzer = CointegrationAnalysis()
    price1_series = pd.Series(price_df[TICKER1])
    price2_series = pd.Series(price_df[TICKER2])
    is_cointegrated = coint_analyzer.test_cointegration(price1_series, price2_series)
    
    hedge_ratio = coint_analyzer.calculate_hedge_ratio(price1_series, price2_series)
    spread = coint_analyzer.calculate_spread(price1_series, price2_series)
    
    spread_analyzer = SpreadAnalysis()
    spread_analyzer.set_spread(spread)
    
    rolling_stats = spread_analyzer.calculate_rolling_statistics(WINDOW_SIZES)
    z_scores = spread_analyzer.calculate_z_scores(WINDOW_SIZES)
    
    strategy = TradingStrategy(price1_series, price2_series, spread, TICKER1, TICKER2)
    
    entry_thresholds_list = ENTRY_THRESHOLDS.tolist()
    exit_thresholds_list = EXIT_THRESHOLDS.tolist()
    
    optimization_results = strategy.optimize_parameters(
        z_scores, WINDOW_SIZES, entry_thresholds_list, exit_thresholds_list
    )
    
    best_params = optimization_results['best_params']
    best_performance = optimization_results['best_performance']
    
    results_str = []
    results_str.append("=" * 60)
    results_str.append("STATISTICAL ARBITRAGE BACKTESTING SYSTEM RESULTS")
    results_str.append("=" * 60)
    results_str.append(f"Trading Pair: {TICKER1} vs {TICKER2}")
    results_str.append(f"Date Range: {START_DATE} to {END_DATE}")
    results_str.append("")
    results_str.append("COINTEGRATION ANALYSIS:")
    results_str.append(f"  Cointegrated: {'Yes' if is_cointegrated else 'No'}")
    results_str.append(f"  Hedge Ratio (β): {hedge_ratio:.4f}")
    results_str.append("")
    results_str.append("OPTIMAL PARAMETERS:")
    results_str.append(f"  Window Size: {best_params['window_size']} days")
    results_str.append(f"  Entry Threshold: {float(best_params['entry_threshold']):.2f}")
    results_str.append(f"  Exit Threshold: {float(best_params['exit_threshold']):.2f}")
    results_str.append(f"  Strategy Type: {best_params['strategy_type']}")
    results_str.append("")
    results_str.append("PERFORMANCE METRICS:")
    results_str.append(f"  Total Return: {best_performance['total_return']:.4f} ({best_performance['total_return']*100:.2f}%)")
    results_str.append(f"  Annualized Return: {best_performance['total_return']*252/len(price_df):.4f} ({best_performance['total_return']*252/len(price_df)*100:.2f}%)")
    results_str.append(f"  Sharpe Ratio: {best_performance['sharpe_ratio']:.4f}")
    results_str.append(f"  Max Drawdown: {best_performance['max_drawdown']:.4f} ({best_performance['max_drawdown']*100:.2f}%)")
    results_str.append(f"  Win Rate: {best_performance['win_rate']:.4f} ({best_performance['win_rate']*100:.2f}%)")
    results_str.append(f"  Volatility: {best_performance['volatility']:.4f} ({best_performance['volatility']*100:.2f}%)")
    results_str.append(f"  Number of Trades: {best_performance['num_trades']}")
    results_str.append("")
    results_str.append("=" * 60)
    results_str.append("END OF RESULTS")
    results_str.append("=" * 60)
    
    print("\n".join(results_str))
    
    with open("results.txt", "w") as f:
        f.write("\n".join(results_str))

if __name__ == "__main__":
    main() 