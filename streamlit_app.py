"""
Streamlit Dashboard for Stock Market Prediction
Interactive web interface for predictions, backtesting, and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append('src')

from src.data_loader import StockDataLoader
from src.feature_engineering import FeatureEngineer
from src.model_train import ModelTrainer
from src.backtester import Backtester

# Try to import LSTM - gracefully handle if TensorFlow not available
try:
    from src.lstm_model import LSTMModelTrainer
    LSTM_AVAILABLE = True
except ImportError as e:
    LSTM_AVAILABLE = False
    print(f"LSTM not available: {e}")


# Page configuration
st.set_page_config(
    page_title="Stock Market Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


# Title and description
st.title("📈 Indian Stock Market Prediction System")
st.markdown("AI/ML powered Indian stock market prediction with backtesting and analysis")


# Sidebar
st.sidebar.header("Configuration")

# Popular Indian stocks
popular_stocks = {
    "Indices": ["^NSEI (Nifty 50)", "^NSEBANK (Bank Nifty)", "^BSESN (Sensex)"],
    "Commodities": ["GC=F (Gold)", "SI=F (Silver)"],
    "Top 10 Large Cap": [
        "RELIANCE.NS (Reliance Industries)",
        "TCS.NS (Tata Consultancy Services)",
        "HDFCBANK.NS (HDFC Bank)",
        "INFY.NS (Infosys)",
        "ICICIBANK.NS (ICICI Bank)",
        "HINDUNILVR.NS (Hindustan Unilever)",
        "ITC.NS (ITC Ltd)",
        "SBIN.NS (State Bank of India)",
        "BHARTIARTL.NS (Bharti Airtel)",
        "KOTAKBANK.NS (Kotak Mahindra Bank)"
    ],
    "IT Sector": [
        "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", 
        "LTI.NS", "LTTS.NS", "MPHASIS.NS", "COFORGE.NS", "PERSISTENT.NS"
    ],
    "Banking": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "INDUSINDBK.NS", "BANDHANBNK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "PNB.NS"
    ],
    "Automobile": [
        "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
        "EICHERMOT.NS", "ASHOKLEY.NS", "TVSMOTOR.NS", "MOTHERSON.NS", "BOSCHLTD.NS"
    ],
    "FMCG": [
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS",
        "MARICO.NS", "GODREJCP.NS", "COLPAL.NS", "TATACONSUM.NS", "EMAMILTD.NS"
    ],
    "Pharma": [
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "AUROPHARMA.NS",
        "LUPIN.NS", "BIOCON.NS", "TORNTPHARM.NS", "ALKEM.NS", "LAURUSLABS.NS"
    ],
    "Energy & Oil": [
        "RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "NTPC.NS",
        "POWERGRID.NS", "ADANIGREEN.NS", "ADANIPOWER.NS", "GAIL.NS", "TATAPOWER.NS"
    ],
    "Metals & Mining": [
        "TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "COALINDIA.NS", "VEDL.NS",
        "JINDALSTEL.NS", "SAIL.NS", "NMDC.NS", "NATIONALUM.NS", "HINDZINC.NS"
    ],
    "Infrastructure": [
        "LT.NS", "ULTRACEMCO.NS", "GRASIM.NS", "ADANIPORTS.NS", "AMBUJACEM.NS",
        "ACC.NS", "DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "BRIGADE.NS"
    ],
    "Telecom & Media": [
        "BHARTIARTL.NS", "INDIAMART.NS", "ZEEL.NS", "HATHWAY.NS", "TATACOMM.NS"
    ],
    "Consumer Durables": [
        "TITAN.NS", "ASIANPAINT.NS", "PIDILITIND.NS", "VOLTAS.NS", "HAVELLS.NS",
        "CROMPTON.NS", "WHIRLPOOL.NS", "BATAINDIA.NS", "RELAXO.NS", "VBL.NS"
    ],
    "Financial Services": [
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS", "ICICIPRULI.NS",
        "CHOLAFIN.NS", "MUTHOOTFIN.NS", "PFC.NS", "RECLTD.NS", "LICHSGFIN.NS"
    ],
    "Adani Group": [
        "ADANIENT.NS", "ADANIPORTS.NS", "ADANIGREEN.NS", "ADANIPOWER.NS", 
        "ADANITRANS.NS", "ATGL.NS"
    ],
    "Tata Group": [
        "TCS.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TATACONSUM.NS", "TITAN.NS",
        "TATAPOWER.NS", "TATACOMM.NS", "VOLTAS.NS", "TRENT.NS", "RALLIS.NS"
    ],
    "Mid Cap Popular": [
        "PAGEIND.NS", "PIIND.NS", "BERGEPAINT.NS", "SRF.NS", "ASTRAL.NS",
        "LALPATHLAB.NS", "DMART.NS", "JUBLFOOD.NS", "MCDOWELL-N.NS", "METROPOLIS.NS"
    ]
}

# Create stock selector
st.sidebar.subheader("📊 Select Stock")
category = st.sidebar.selectbox("Category", list(popular_stocks.keys()))
stock_symbol = st.sidebar.selectbox("Stock/Index", popular_stocks[category])

# Extract ticker from display name
if "(" in stock_symbol:
    ticker = stock_symbol.split(" (")[0]
else:
    ticker = stock_symbol

# Manual input option
st.sidebar.markdown("---")
manual_ticker = st.sidebar.text_input("Or enter ticker manually", value="", help="e.g., RELIANCE.NS, TCS.NS, ^NSEI")
if manual_ticker:
    ticker = manual_ticker

st.sidebar.caption(f"🎯 Selected: **{ticker}**")
st.sidebar.markdown("---")

# Indian market info
with st.sidebar.expander("ℹ️ Indian Market Info"):
    st.markdown("""
    **NSE Tickers Format:** `SYMBOL.NS`  
    **BSE Tickers Format:** `SYMBOL.BO`  
    **Indices:** Use `^` prefix
    
    **Examples:**
    - Reliance: `RELIANCE.NS`
    - Nifty 50: `^NSEI`
    - Bank Nifty: `^NSEBANK`
    - Sensex: `^BSESN`
    - Gold: `GC=F`
    - Silver: `SI=F`
    
    **Note:** Prices shown in ₹ (INR)
    """)
years = st.sidebar.slider("Years of Historical Data", min_value=1, max_value=10, value=5)

model_options = ["XGBoost", "Random Forest", "Logistic Regression"]
if LSTM_AVAILABLE:
    model_options.append("LSTM")

model_type = st.sidebar.selectbox(
    "Select Model",
    model_options,
    help="Choose the prediction model"
)

# Map display names to internal names
model_map = {
    "XGBoost": "xgboost",
    "Random Forest": "random_forest",
    "Logistic Regression": "logistic_regression",
    "LSTM": "lstm"
}


# Session state for caching
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False


# Load data button
if st.sidebar.button("Load Data", type="primary"):
    with st.spinner(f"Loading {ticker} data..."):
        try:
            # Load data
            loader = StockDataLoader(ticker.upper(), years=years)
            data = loader.download_data(save_to_csv=False)
            data = loader.create_targets(data)
            
            # Create features
            fe = FeatureEngineer(data)
            data_with_features = fe.add_all_features()
            
            # Store in session state
            st.session_state.data = data_with_features
            st.session_state.loader = loader
            st.session_state.fe = fe
            st.session_state.data_loaded = True
            st.session_state.ticker = ticker.upper()
            
            st.sidebar.success(f"✅ Data loaded successfully! ({len(data_with_features)} rows)")
        except Exception as e:
            st.sidebar.error(f"Error loading data: {str(e)}")


# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🤖 Prediction", "📉 Backtest", "📈 Charts", "🔍 Features"])


# Tab 1: Overview
with tab1:
    st.header("Data Overview")
    
    if st.session_state.data_loaded:
        data = st.session_state.data
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", len(data))
        with col2:
            st.metric("Total Features", len(data.columns))
        with col3:
            current_price = data['close'].iloc[-1]
            st.metric("Current Price", f"₹{current_price:,.2f}")
        with col4:
            price_change = ((data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2] * 100)
            st.metric("Daily Change", f"{price_change:.2f}%", delta=f"{price_change:.2f}%")
        
        # Additional market statistics
        st.markdown("---")
        st.subheader("📊 Market Statistics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            high_52w = data['close'].tail(252).max() if len(data) >= 252 else data['close'].max()
            st.metric("52-Week High", f"₹{high_52w:,.2f}")
        
        with col2:
            low_52w = data['close'].tail(252).min() if len(data) >= 252 else data['close'].min()
            st.metric("52-Week Low", f"₹{low_52w:,.2f}")
        
        with col3:
            avg_volume = data['volume'].tail(30).mean()
            st.metric("Avg Volume (30D)", f"{avg_volume:,.0f}")
        
        with col4:
            volatility = data['close'].pct_change().tail(30).std() * 100
            st.metric("Volatility (30D)", f"{volatility:.2f}%")
        
        with col5:
            ytd_return = ((data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * 100)
            st.metric("Total Return", f"{ytd_return:+.2f}%")
        
        st.subheader("Recent Price Data (Last 10 Days)")
        # Format the dataframe for better readability
        recent_data = data[['open', 'high', 'low', 'close', 'volume']].tail(10).copy()
        
        # Add additional helpful columns
        recent_data['Day Change (%)'] = ((recent_data['close'] - recent_data['open']) / recent_data['open'] * 100)
        recent_data['Range (₹)'] = recent_data['high'] - recent_data['low']
        
        # Format for display
        display_data = recent_data.copy()
        display_data['open'] = recent_data['open'].apply(lambda x: f"₹{x:,.2f}")
        display_data['high'] = recent_data['high'].apply(lambda x: f"₹{x:,.2f}")
        display_data['low'] = recent_data['low'].apply(lambda x: f"₹{x:,.2f}")
        display_data['close'] = recent_data['close'].apply(lambda x: f"₹{x:,.2f}")
        display_data['volume'] = recent_data['volume'].apply(lambda x: f"{x:,.0f}")
        display_data['Day Change (%)'] = recent_data['Day Change (%)'].apply(lambda x: f"{x:+.2f}%")
        display_data['Range (₹)'] = recent_data['Range (₹)'].apply(lambda x: f"₹{x:,.2f}")
        
        display_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Day Change', 'Intraday Range']
        
        # Style the dataframe
        st.dataframe(display_data, use_container_width=True)
        
        # Price chart
        st.subheader("Price History")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        ))
        fig.update_layout(
            title=f"{st.session_state.ticker} Price Chart",
            yaxis_title="Price (₹)",
            xaxis_title="Date",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("👈 Please load data from the sidebar to get started")


# Tab 2: Prediction
with tab2:
    st.header("Next-Day Prediction")
    
    if st.session_state.data_loaded:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Train Model & Predict", type="primary"):
                with st.spinner(f"Training {model_type} model..."):
                    try:
                        data = st.session_state.data
                        model_name = model_map[model_type]
                        
                        if model_name in ['logistic_regression', 'random_forest', 'xgboost']:
                            # Train ML model
                            trainer = ModelTrainer()
                            X_train, X_test, y_train, y_test = trainer.prepare_data(data)
                            
                            if model_name == 'xgboost':
                                trainer.train_xgboost(X_train, y_train)
                            elif model_name == 'random_forest':
                                trainer.train_random_forest(X_train, y_train)
                            else:
                                trainer.train_logistic_regression(X_train, y_train)
                            
                            # Evaluate
                            metrics = trainer.evaluate_model(model_name, X_test, y_test)
                            
                            # Get prediction for latest data
                            latest_data = data.iloc[-1:]
                            predictions, probabilities = trainer.predict(model_name, latest_data)
                            
                            prediction = int(predictions[0])
                            probability = float(probabilities[0][1])
                            
                            st.session_state.trainer = trainer
                            
                        else:  # LSTM
                            trainer = LSTMModelTrainer(sequence_length=60)
                            X_train, X_test, y_train, y_test = trainer.prepare_sequences(data)
                            
                            trainer.build_lstm_model(lstm_units=[128, 64])
                            trainer.train_model(X_train, y_train, X_test, y_test, epochs=30, verbose=0)
                            
                            metrics = trainer.evaluate_model(X_test, y_test)
                            
                            # Get prediction
                            result = trainer.predict_next_day(data)
                            prediction = 1 if result['prediction'] == 'UP' else 0
                            probability = result['probability']
                            
                            st.session_state.lstm_trainer = trainer
                        
                        # Store results
                        st.session_state.prediction = prediction
                        st.session_state.probability = probability
                        st.session_state.metrics = metrics
                        st.session_state.model_trained = True
                        
                        st.success("✅ Model trained successfully!")
                        
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        
        with col2:
            if st.session_state.model_trained:
                st.subheader("Model Performance")
                metrics = st.session_state.metrics
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                col2.metric("F1 Score", f"{metrics['f1']:.2%}")
                col3.metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")
        
        if st.session_state.model_trained:
            st.markdown("---")
            st.subheader("Prediction Result")
            
            prediction = st.session_state.prediction
            probability = st.session_state.probability
            confidence = abs(probability - 0.5) * 2
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.success("### 📈 UP")
                    st.write("Next day prediction: **Price will increase**")
                else:
                    st.error("### 📉 DOWN")
                    st.write("Next day prediction: **Price will decrease**")
            
            with col2:
                st.metric("Probability", f"{probability:.2%}")
                st.progress(probability)
            
            with col3:
                st.metric("Confidence", f"{confidence:.2%}")
                if confidence > 0.5:
                    st.write("🟢 High confidence")
                elif confidence > 0.3:
                    st.write("🟡 Medium confidence")
                else:
                    st.write("🔴 Low confidence")
            
    else:
        st.info("👈 Please load data first")


# Tab 3: Backtest
with tab3:
    st.header("Strategy Backtesting")
    
    if st.session_state.data_loaded and st.session_state.model_trained:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            initial_capital = st.number_input("Initial Capital (₹)", min_value=10000, max_value=10000000, value=1000000, step=10000)
        with col2:
            transaction_cost = st.number_input("Transaction Cost (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01) / 100
        with col3:
            position_size = st.slider("Position Size (%)", min_value=10, max_value=100, value=100) / 100
        
        if st.button("Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                try:
                    data = st.session_state.data
                    model_name = model_map[model_type]
                    
                    # Get predictions for test set
                    if model_name in ['logistic_regression', 'random_forest', 'xgboost']:
                        trainer = st.session_state.trainer
                        X_train, X_test, y_train, y_test = trainer.prepare_data(data)
                        predictions, _ = trainer.predict(model_name, X_test)
                        test_data = data.iloc[len(X_train):]
                    else:
                        trainer = st.session_state.lstm_trainer
                        X_train, X_test, y_train, y_test = trainer.prepare_sequences(data)
                        predictions, _ = trainer.predict(X_test)
                        test_data = data.iloc[trainer.sequence_length + len(X_train):]
                    
                    # Run backtest
                    backtester = Backtester(
                        initial_capital=initial_capital,
                        transaction_cost=transaction_cost,
                        position_size=position_size
                    )
                    
                    results = backtester.run_backtest(test_data, predictions)
                    metrics = backtester.calculate_metrics()
                    comparison = backtester.compare_with_buy_and_hold(test_data)
                    
                    st.session_state.backtest_results = results
                    st.session_state.backtest_metrics = metrics
                    st.session_state.backtest_comparison = comparison
                    
                    st.success("✅ Backtest completed!")
                    
                except Exception as e:
                    st.error(f"Error running backtest: {str(e)}")
        
        if 'backtest_metrics' in st.session_state:
            st.markdown("---")
            st.subheader("Performance Metrics")
            
            metrics = st.session_state.backtest_metrics
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return", f"{metrics['Total Return (%)']:.2f}%")
            col2.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
            col3.metric("Max Drawdown", f"{metrics['Max Drawdown (%)']:.2f}%")
            col4.metric("Win Rate", f"{metrics['Win Rate (%)']:.2f}%")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Annual Return", f"{metrics['Annualized Return (%)']:.2f}%")
            col2.metric("Volatility", f"{metrics['Volatility (%)']:.2f}%")
            col3.metric("Calmar Ratio", f"{metrics['Calmar Ratio']:.2f}")
            col4.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}")
            
            st.markdown("---")
            st.subheader("Strategy vs Buy & Hold")
            
            comparison = st.session_state.backtest_comparison
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Strategy Return", f"{comparison['Strategy Return (%)']:.2f}%")
            col2.metric("Buy & Hold Return", f"{comparison['Buy & Hold Return (%)']:.2f}%")
            col3.metric("Outperformance", f"{comparison['Outperformance (%)']:.2f}%")
            
            # Equity curve
            st.subheader("Equity Curve")
            results = st.session_state.backtest_results
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results.index, y=results['equity'], 
                                    mode='lines', name='Strategy Equity',
                                    line=dict(color='blue', width=2)))
            fig.add_hline(y=initial_capital, line_dash="dash", 
                         line_color="red", annotation_text="Initial Capital")
            fig.update_layout(
                title="Equity Curve Over Time",
                xaxis_title="Date",
                yaxis_title="Equity (₹)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        if not st.session_state.data_loaded:
            st.info("👈 Please load data first")
        else:
            st.info("👆 Please train a model first in the Prediction tab")


# Tab 4: Charts
with tab4:
    st.header("Technical Analysis Charts")
    
    if st.session_state.data_loaded:
        data = st.session_state.data
        
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Price with Indicators", "Volume", "RSI", "MACD", "Bollinger Bands"]
        )
        
        if chart_type == "Price with Indicators":
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              vertical_spacing=0.03, row_heights=[0.7, 0.3])
            
            # Price and moving averages
            fig.add_trace(go.Candlestick(x=data.index, open=data['open'], high=data['high'],
                                        low=data['low'], close=data['close'], name='Price'),
                         row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['sma_20'], name='SMA 20',
                                    line=dict(color='orange', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['sma_50'], name='SMA 50',
                                    line=dict(color='blue', width=1)), row=1, col=1)
            
            # Volume
            fig.add_trace(go.Bar(x=data.index, y=data['volume'], name='Volume',
                                marker_color='lightblue'), row=2, col=1)
            
            fig.update_layout(height=600, title_text="Price Chart with Moving Averages")
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Volume":
            fig = go.Figure()
            fig.add_trace(go.Bar(x=data.index, y=data['volume'], name='Volume'))
            fig.update_layout(title="Volume Chart", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "RSI":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['rsi'], name='RSI',
                                    line=dict(color='purple', width=2)))
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig.update_layout(title="RSI Indicator", yaxis_title="RSI", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "MACD":
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03)
            
            fig.add_trace(go.Scatter(x=data.index, y=data['close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['macd'], name='MACD'), row=2, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['macd_signal'], name='Signal'), row=2, col=1)
            fig.add_trace(go.Bar(x=data.index, y=data['macd_diff'], name='Histogram'), row=2, col=1)
            
            fig.update_layout(height=600, title_text="MACD Indicator")
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Bollinger Bands":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['bb_upper'], name='Upper Band',
                                    line=dict(color='red', width=1)))
            fig.add_trace(go.Scatter(x=data.index, y=data['bb_middle'], name='Middle Band',
                                    line=dict(color='blue', width=1)))
            fig.add_trace(go.Scatter(x=data.index, y=data['bb_lower'], name='Lower Band',
                                    line=dict(color='green', width=1)))
            fig.add_trace(go.Scatter(x=data.index, y=data['close'], name='Close Price',
                                    line=dict(color='black', width=2)))
            fig.update_layout(title="Bollinger Bands", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.info("👈 Please load data first")


# Tab 5: Features
with tab5:
    st.header("Feature Analysis")
    
    if st.session_state.data_loaded and st.session_state.model_trained:
        model_name = model_map[model_type]
        
        if model_name in ['logistic_regression', 'random_forest', 'xgboost']:
            trainer = st.session_state.trainer
            
            st.subheader("Top 20 Most Important Features")
            
            feature_importance = trainer.get_feature_importance(model_name, top_n=20)
            
            if feature_importance is not None:
                fig = px.bar(feature_importance, x='importance', y='feature',
                           orientation='h', title=f'Feature Importance ({model_type})')
                fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Feature Importance Table")
                st.dataframe(feature_importance, use_container_width=True)
        else:
            st.info("Feature importance is only available for tree-based models (XGBoost, Random Forest)")
        
        st.markdown("---")
        st.subheader("All Features")
        data = st.session_state.data
        st.write(f"Total features: {len(data.columns)}")
        
        feature_groups = st.session_state.fe.get_feature_importance_groups()
        
        for group_name, features in feature_groups.items():
            with st.expander(f"{group_name.upper()} ({len(features)} features)"):
                st.write(features)
                
    else:
        if not st.session_state.data_loaded:
            st.info("👈 Please load data first")
        else:
            st.info("👆 Please train a model first in the Prediction tab")


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Stock Market Prediction System | Built with Streamlit, scikit-learn, XGBoost, TensorFlow</p>
    </div>
    """,
    unsafe_allow_html=True
)
