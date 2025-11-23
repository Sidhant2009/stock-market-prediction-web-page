"""
Main Training Script
Train all models and generate predictions
"""

import sys
import os
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import StockDataLoader
from src.feature_engineering import FeatureEngineer
from src.model_train import ModelTrainer
from src.lstm_model import LSTMModelTrainer
from src.backtester import Backtester
from src.utils import save_results_to_json, generate_report


def main(ticker: str, years: int = 5, train_lstm: bool = True):
    """
    Main training pipeline
    
    Args:
        ticker: Stock ticker symbol
        years: Years of historical data
        train_lstm: Whether to train LSTM model
    """
    print("="*70)
    print(f"STOCK MARKET PREDICTION SYSTEM")
    print(f"Ticker: {ticker}")
    print(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Step 1: Load Data
    print("\n[1/6] Loading data...")
    loader = StockDataLoader(ticker, years=years)
    data = loader.download_data()
    data = loader.create_targets(data)
    
    print(f"Data loaded: {len(data)} rows")
    
    # Step 2: Feature Engineering
    print("\n[2/6] Creating features...")
    fe = FeatureEngineer(data)
    data_with_features = fe.add_all_features()
    
    print(f"Features created: {len(data_with_features.columns)} features")
    
    # Step 3: Train ML Models
    print("\n[3/6] Training ML models...")
    ml_trainer = ModelTrainer(scale_method='standard')
    X_train, X_test, y_train, y_test = ml_trainer.prepare_data(data_with_features)
    
    # Train all ML models
    ml_trainer.train_all_models(X_train, y_train)
    
    # Evaluate
    results_df = ml_trainer.evaluate_all_models(X_test, y_test)
    
    # Save models
    ml_trainer.save_models()
    
    print("\nML Models Performance:")
    print(results_df.to_string(index=False))
    
    # Step 4: Train LSTM Model (optional)
    if train_lstm:
        print("\n[4/6] Training LSTM model...")
        lstm_trainer = LSTMModelTrainer(sequence_length=60)
        X_train_seq, X_test_seq, y_train_seq, y_test_seq = lstm_trainer.prepare_sequences(data_with_features)
        
        # Build and train
        lstm_trainer.build_lstm_model(lstm_units=[128, 64], dropout_rate=0.3)
        lstm_trainer.train_model(X_train_seq, y_train_seq, X_test_seq, y_test_seq, 
                                epochs=50, batch_size=32, verbose=1)
        
        # Evaluate
        lstm_metrics = lstm_trainer.evaluate_model(X_test_seq, y_test_seq)
        
        # Save
        lstm_trainer.save_model('models/lstm_model.h5')
    else:
        print("\n[4/6] Skipping LSTM training...")
    
    # Step 5: Backtest Best ML Model (XGBoost)
    print("\n[5/6] Running backtest with XGBoost...")
    predictions, _ = ml_trainer.predict('xgboost', X_test)
    test_data = data_with_features.iloc[len(X_train):]
    
    backtester = Backtester(initial_capital=100000, transaction_cost=0.001)
    results = backtester.run_backtest(test_data, predictions)
    
    # Calculate metrics
    backtest_metrics = backtester.calculate_metrics()
    backtester.print_metrics()
    
    # Compare with buy and hold
    comparison = backtester.compare_with_buy_and_hold(test_data)
    
    # Step 6: Generate Reports
    print("\n[6/6] Generating reports...")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    # Model results
    save_results_to_json(
        ml_trainer.results['xgboost'],
        f'results/{ticker}_xgboost_metrics.json'
    )
    
    # Backtest results
    save_results_to_json(
        backtest_metrics,
        f'results/{ticker}_backtest_metrics.json'
    )
    
    # Generate comprehensive report
    generate_report(
        ticker=ticker,
        model_name='XGBoost',
        metrics=ml_trainer.results['xgboost'],
        backtest_metrics=backtest_metrics,
        save_path=f'results/{ticker}_report.txt'
    )
    
    # Feature importance
    print("\nTop 20 Features (XGBoost):")
    importance = ml_trainer.get_feature_importance('xgboost', top_n=20)
    print(importance.to_string(index=False))
    importance.to_csv(f'results/{ticker}_feature_importance.csv', index=False)
    
    # Save trades
    trades_df = backtester.get_trades_summary()
    if len(trades_df) > 0:
        trades_df.to_csv(f'results/{ticker}_trades.csv', index=False)
        print(f"\nTrades saved to results/{ticker}_trades.csv")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"Models saved to: models/")
    print(f"Results saved to: results/")
    print("="*70)
    
    return {
        'ml_trainer': ml_trainer,
        'lstm_trainer': lstm_trainer if train_lstm else None,
        'backtester': backtester,
        'results': results_df,
        'backtest_metrics': backtest_metrics
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train stock prediction models')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--years', type=int, default=5, help='Years of historical data')
    parser.add_argument('--no-lstm', action='store_true', help='Skip LSTM training')
    
    args = parser.parse_args()
    
    main(
        ticker=args.ticker,
        years=args.years,
        train_lstm=not args.no_lstm
    )
