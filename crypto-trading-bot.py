"""
Crypto Trading Bot - Optimized for NVIDIA RTX 3080 & Intel i9-13900K
================================================================
A high-performance cryptocurrency trading bot with AI/ML capabilities
and hardware optimization for maximum efficiency.

Features:
- Deep Learning prediction model (LSTM) with GPU acceleration
- Advanced technical indicators and sentiment analysis
- Risk management with adaptive position sizing
- Multi-asset trading support
- Interactive user interface
- Performance monitoring and visualization

Hardware Optimization:
- NVIDIA RTX 3080 with CUDA acceleration
- Intel i9-13900K multi-threading support
- 32GB RAM management
- Multi-monitor display support

Author: BeoWulf5011
Date: 2025-02-26
Version: 1.2.0
License: MIT
"""

# Standard Libraries
import os
import sys
import time
import json
import logging
import datetime
import threading
from typing import Dict, List, Tuple, Optional, Union, Any

# Data Science & ML Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# Trading & API Libraries
import ccxt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CryptoTradingBot")

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Technical indicator parameters
INDICATOR_PARAMS = {
    'sma_periods': [7, 14, 25, 50, 99, 200],
    'ema_periods': [8, 12, 21, 26, 34, 55, 89],
    'rsi_periods': [9, 14, 21],
    'bb_periods': [20, 50],
    'stoch_periods': [(14, 3), (21, 5)],
}

# ML model configuration
ML_CONFIG = {
    'lstm_units': [128, 256, 128],
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'batch_size': 64,
    'epochs': 100,
}

# Exchange configuration
EXCHANGE_CONFIG = {
    'retry_attempts': 3,
    'backoff_factor': 2,
    'default_min_notional': 10.0,
    'default_timeframe': '1h',
}

# Trading parameters
TRADING_PARAMS = {
    'default_trade_amount': 0.02,  # 2% of balance
    'default_stop_loss': 0.02,     # 2% 
    'default_take_profit': 0.05,   # 5%
    'min_quote_reserve': 5.0,      # Minimum reserve in quote currency
    'high_volatility_threshold': 0.8,
    'med_volatility_threshold': 0.5,
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safely perform division to avoid division by zero errors."""
    return a / b if b and b != 0 else default

def format_price(price: float, precision: int = 8) -> str:
    """Format price with appropriate precision."""
    if price > 1000:
        return f"{price:.2f}"
    elif price > 1:
        return f"{price:.4f}"
    else:
        return f"{price:.{precision}f}"

def timestamp_to_str(timestamp) -> str:
    """Convert timestamp to formatted string."""
    if isinstance(timestamp, (int, float)):
        dt = datetime.datetime.fromtimestamp(timestamp / 1000) if timestamp > 1e10 else datetime.datetime.fromtimestamp(timestamp)
    else:
        dt = timestamp
    return dt.strftime("%Y-%m-%d %H:%M:%S")

# =============================================================================
# GPU ACCELERATOR CLASS
# =============================================================================

class GPUAccelerator:
    """
    Manages GPU acceleration for TensorFlow models.
    Optimized for NVIDIA RTX 3080.
    """
    
    def __init__(self):
        self.gpu_available = False
        self.gpu_info = None
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize GPU settings for optimal performance."""
        try:
            # Try to get available physical devices
            physical_devices = tf.config.list_physical_devices('GPU')
            
            if len(physical_devices) > 0:
                # Configure memory growth to avoid allocating all memory at once
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                
                # Enable mixed precision for better performance
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                
                # Get GPU information
                self.gpu_info = {
                    'device': physical_devices[0],
                    'name': physical_devices[0].name,
                    'type': 'GPU'
                }
                
                # Try to get detailed GPU info
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    
                    # Get memory information
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.gpu_info['memory_total'] = meminfo.total / (1024**2) # MB
                    
                    # Get compute capability
                    cc_major, cc_minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    self.gpu_info['compute_capability'] = f"{cc_major}.{cc_minor}"
                    
                    pynvml.nvmlShutdown()
                except (ImportError, Exception) as e:
                    logger.warning(f"Could not get detailed GPU info: {e}")
                
                self.gpu_available = True
                logger.info(f"GPU acceleration enabled on {self.gpu_info['name']}")
                
                if 'RTX 3080' in str(physical_devices[0]):
                    logger.info("NVIDIA RTX 3080 detected - using optimized configuration")
                
                self.initialized = True
                return True
            else:
                logger.warning("No GPU detected, using CPU for computations")
                self.gpu_available = False
                self.initialized = True
                return False
        
        except Exception as e:
            logger.error(f"Error initializing GPU: {e}")
            self.gpu_available = False
            self.initialized = True
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current GPU status."""
        if not self.initialized:
            self.initialize()
        
        status = {
            'available': self.gpu_available,
            'info': self.gpu_info
        }
        
        # Try to get current usage
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Memory usage
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            status['memory_used'] = meminfo.used / (1024**2)  # MB
            status['memory_free'] = meminfo.free / (1024**2)  # MB
            status['memory_total'] = meminfo.total / (1024**2)  # MB
            
            # GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            status['gpu_utilization'] = utilization.gpu
            status['memory_utilization'] = utilization.memory
            
            # Temperature
            status['temperature'] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            pynvml.nvmlShutdown()
        except (ImportError, Exception) as e:
            status['error'] = str(e)
        
        return status

    def optimize_for_training(self, batch_size: int = 64) -> Dict[str, Any]:
        """Get optimized parameters for model training."""
        if not self.initialized:
            self.initialize()
        
        # Default parameters (CPU)
        params = {
            'batch_size': 32,
            'use_mixed_precision': False,
            'compile_params': {
                'optimizer': 'adam'
            }
        }
        
        if self.gpu_available:
            # Adjust batch size based on available GPU memory
            if self.gpu_info and 'memory_total' in self.gpu_info:
                # RTX 3080 optimization (10GB VRAM)
                if self.gpu_info['memory_total'] >= 9000:  # ~9GB+ VRAM
                    params['batch_size'] = batch_size
                    params['use_mixed_precision'] = True
                    params['compile_params'] = {
                        'optimizer': tf.keras.optimizers.Adam(learning_rate=ML_CONFIG['learning_rate'])
                    }
            else:
                # Generic GPU optimization
                params['batch_size'] = batch_size
                params['use_mixed_precision'] = True
        
        return params


# =============================================================================
# TECHNICAL INDICATOR CALCULATOR
# =============================================================================

class TechnicalIndicators:
    """
    Calculator for technical indicators used in trading decisions.
    Optimized for fast computation with pandas and numpy operations.
    """
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to the dataframe."""
        if df is None or len(df) < 30:
            logger.warning(f"Insufficient data for technical indicators: {len(df) if df is not None else 'None'}")
            return df
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Check for NaN values and handle them
        if df.isna().sum().sum() > 0:
            logger.warning(f"NaN values detected in dataframe, filling with forward/backward fill")
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Add each group of indicators
        df = TechnicalIndicators.add_moving_averages(df)
        df = TechnicalIndicators.add_oscillators(df)
        df = TechnicalIndicators.add_volatility_indicators(df)
        df = TechnicalIndicators.add_volume_indicators(df)
        df = TechnicalIndicators.add_trend_indicators(df)
        
        # Fill any remaining NaN values
        df = df.fillna(method='bfill').fillna(0)
        
        return df
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Add Simple and Exponential Moving Averages."""
        # Simple Moving Averages (SMA)
        for period in INDICATOR_PARAMS['sma_periods']:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Exponential Moving Averages (EMA)
        for period in INDICATOR_PARAMS['ema_periods']:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Alternative faster MACD
        df['macd_fast'] = df['ema_8'] - df['ema_21']
        df['macd_fast_signal'] = df['macd_fast'].ewm(span=5, adjust=False).mean()
        df['macd_fast_hist'] = df['macd_fast'] - df['macd_fast_signal']
        
        return df
    
    @staticmethod
    def add_oscillators(df: pd.DataFrame) -> pd.DataFrame:
        """Add oscillator indicators (RSI, Stochastic, etc.)."""
        # RSI - Relative Strength Index
        for period in INDICATOR_PARAMS['rsi_periods']:
            delta = df['close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # Avoid division by zero
            rs = avg_gain / avg_loss.replace(0, 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # For backward compatibility, set the standard RSI
        df['rsi'] = df['rsi_14']
        
        # Stochastic Oscillator
        for k_period, d_period in INDICATOR_PARAMS['stoch_periods']:
            low_min = df['low'].rolling(window=k_period).min()
            high_max = df['high'].rolling(window=k_period).max()
            
            # Avoid division by zero
            denom = high_max - low_min
            denom = denom.replace(0, 1e-10)
            
            df[f'stoch_k_{k_period}'] = 100 * ((df['close'] - low_min) / denom)
            df[f'stoch_d_{k_period}_{d_period}'] = df[f'stoch_k_{k_period}'].rolling(window=d_period).mean()
        
        # For backward compatibility
        df['stoch_k'] = df['stoch_k_14']
        df['stoch_d'] = df['stoch_d_14_3']
        
        # Relative Volatility Index (RVI)
        std_14 = df['close'].rolling(window=14).std()
        std_change = std_14.diff()
        std_up = std_change.clip(lower=0)
        std_down = -std_change.clip(upper=0)
        
        std_up_avg = std_up.rolling(window=14).mean()
        std_down_avg = std_down.rolling(window=14).mean()
        
        # Avoid division by zero
        denom = std_up_avg + std_down_avg
        denom = denom.replace(0, 1e-10)
        
        df['rvi'] = 100 * std_up_avg / denom
        
        return df
    
    @staticmethod
    def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators (Bollinger Bands, ATR, etc.)."""
        # Bollinger Bands
        for period in INDICATOR_PARAMS['bb_periods']:
            df[f'bb_middle_{period}'] = df['close'].rolling(window=period).mean()
            df[f'bb_std_{period}'] = df['close'].rolling(window=period).std()
            
            for std_dev in [2, 3]:
                df[f'bb_upper_{period}_{std_dev}'] = df[f'bb_middle_{period}'] + std_dev * df[f'bb_std_{period}']
                df[f'bb_lower_{period}_{std_dev}'] = df[f'bb_middle_{period}'] - std_dev * df[f'bb_std_{period}']
        
        # For backward compatibility
        df['bb_middle'] = df['bb_middle_20'] 
        df['bb_std'] = df['bb_std_20']
        df['bb_upper'] = df['bb_upper_20_2']
        df['bb_lower'] = df['bb_lower_20_2']
        
        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(window=14).mean()
        
        # Volatility calculation (historical)
        df['volatility_14'] = df['close'].rolling(window=14).std() / df['close'].rolling(window=14).mean()
        
        return df
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators (OBV, VWAP, etc.)."""
        # Volume Moving Average
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        
        # Volume Ratio
        df['volume_ratio'] = df['volume'] / df['volume_sma_20'].replace(0, 1e-10)
        
        # On-Balance Volume (OBV)
        close_diff = df['close'].diff()
        obv = pd.Series(0, index=df.index)
        
        # Vectorized calculation if possible
        if 'volume' in df:
            obv = (np.sign(close_diff) * df['volume']).fillna(0).cumsum()
        
        df['obv'] = obv
        
        # Ease of Movement (EOM)
        distance_moved = ((df['high'] + df['low']) / 2) - ((df['high'].shift() + df['low'].shift()) / 2)
        box_ratio = (df['volume'] / 100000000) / ((df['high'] - df['low']))
        df['eom_14'] = (distance_moved / box_ratio.replace(0, 1e-10)).rolling(window=14).mean()
        
        return df
    
    @staticmethod
    def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based indicators (ADX, Ichimoku, etc.)."""
        # Average Directional Index (ADX)
        high_diff = df['high'].diff()
        low_diff = df['low'].diff()
        
        plus_dm = np.where((high_diff > 0) & (high_diff > low_diff), high_diff, 0)
        minus_dm = np.where((low_diff > 0) & (low_diff > high_diff.abs()), low_diff, 0)
        
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift()).abs()
        tr3 = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr_period = 14
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/atr_period).mean() / tr.ewm(alpha=1/atr_period).mean().replace(0, 1e-10)
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/atr_period).mean() / tr.ewm(alpha=1/atr_period).mean().replace(0, 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10).abs()
        
        df['adx'] = dx.ewm(alpha=1/atr_period).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # Ichimoku Cloud
        df['tenkan_sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
        df['kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        df['senkou_span_b'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
        df['chikou_span'] = df['close'].shift(-26)
        
        # Guppy Multiple Moving Averages (GMMA)
        # Short-term EMAs
        for span in [3, 5, 8, 10, 12, 15]:
            df[f'gmma_short_{span}'] = df['close'].ewm(span=span).mean()
        
        # Long-term EMAs
        for span in [30, 35, 40, 45, 50, 60]:
            df[f'gmma_long_{span}'] = df['close'].ewm(span=span).mean()
        
        return df


# =============================================================================
# MARKET SENTIMENT ANALYZER
# =============================================================================

class MarketSentimentAnalyzer:
    """
    Analyzes market sentiment using technical indicators and machine learning.
    Can incorporate external sources like news and social media sentiment.
    """
    
    def __init__(self):
        self.ml_model = None
        self.sentiment_weights = {
            # Trend indicators
            'trend_sma': 0.7,
            'trend_ema': 0.7,
            'price_vs_sma99': 0.5,
            'adx_trend': 0.6,
            
            # Momentum indicators
            'rsi': 0.8,
            'macd': 0.8,
            'macd_hist': 0.6,
            
            # Volatility indicators
            'bollinger': 0.7,
            'bb_width': 0.3,
            
            # Oscillators
            'stochastic_k': 0.6,
            'stochastic_d': 0.5,
            'stoch_cross': 0.4,
            
            # Volume and other
            'volume': 0.5,
            'price_24h': 0.6,
            'ml_prediction': 0.9  # ML prediction has high weight
        }
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market sentiment based on technical indicators."""
        if df is None or len(df) < 2:
            logger.warning("Insufficient data for sentiment analysis")
            return None
        
        try:
            # Get the most recent data point
            current = df.iloc[-1]
            
            # Calculate technical indicator signals
            signals = self._calculate_indicator_signals(df, current)
            
            # Add ML-based prediction if available
            signals = self._add_ml_prediction(df, signals)
            
            # Calculate volatility measure
            volatility = self._calculate_volatility(df)
            
            # Calculate weighted sentiment score
            bullish_score, bearish_score, strength, overall = self._calculate_weighted_score(signals)
            
            # Compile the sentiment analysis result
            return {
                'signals': signals,
                'overall': overall,
                'strength': strength,
                'bullish_score': bullish_score,
                'bearish_score': bearish_score,
                'volatility': volatility,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return None
    
    def _calculate_indicator_signals(self, df: pd.DataFrame, current) -> Dict[str, str]:
        """Calculate signals from various technical indicators."""
        signals = {}
        
        # Trend indicators
        signals['trend_sma'] = 'bullish' if current.get('sma_7', 0) > current.get('sma_25', 0) else 'bearish'
        signals['trend_ema'] = 'bullish' if current.get('ema_12', 0) > current.get('ema_26', 0) else 'bearish'
        signals['price_vs_sma99'] = 'bullish' if current.get('close', 0) > current.get('sma_99', 0) else 'bearish'
        
        # ADX trend strength
        adx_value = current.get('adx', 0)
        if adx_value > 25:
            if current.get('plus_di', 0) > current.get('minus_di', 0):
                signals['adx_trend'] = 'bullish'
            else:
                signals['adx_trend'] = 'bearish'
        else:
            signals['adx_trend'] = 'neutral'
        
        # Momentum indicators
        rsi_value = current.get('rsi', 50)
        if rsi_value > 70:
            signals['rsi'] = 'overbought'
        elif rsi_value < 30:
            signals['rsi'] = 'oversold'
        else:
            signals['rsi'] = 'neutral'
        
        signals['macd'] = 'bullish' if current.get('macd', 0) > current.get('macd_signal', 0) else 'bearish'
        signals['macd_hist'] = 'bullish' if current.get('macd_hist', 0) > 0 else 'bearish'
        
        # Volatility indicators
        bb_upper = current.get('bb_upper', float('inf'))
        bb_lower = current.get('bb_lower', 0)
        close = current.get('close', 0)
        
        if close > bb_upper:
            signals['bollinger'] = 'overbought'
        elif close < bb_lower:
            signals['bollinger'] = 'oversold'
        else:
            signals['bollinger'] = 'neutral'
        
        # BB width shows volatility expansion/contraction
        bb_middle = current.get('bb_middle', 1)  # Default to 1 to avoid division by zero
        if bb_middle == 0:
            bb_middle = 1
            
        bb_width = (current.get('bb_upper', 0) - current.get('bb_lower', 0)) / bb_middle
        signals['bb_width'] = 'expanding' if bb_width > 0.05 else 'contracting'
        
        # Oscillators
        stoch_k = current.get('stoch_k', 50)
        stoch_d = current.get('stoch_d', 50)
        
        if stoch_k > 80:
            signals['stochastic_k'] = 'overbought'
        elif stoch_k < 20:
            signals['stochastic_k'] = 'oversold'
        else:
            signals['stochastic_k'] = 'neutral'
            
        if stoch_d > 80:
            signals['stochastic_d'] = 'overbought'
        elif stoch_d < 20:
            signals['stochastic_d'] = 'oversold'
        else:
            signals['stochastic_d'] = 'neutral'
            
        signals['stoch_cross'] = 'bullish' if stoch_k > stoch_d else 'bearish'
        
        # Volume analysis
        volume = current.get('volume', 0)
        volume_ma = current.get('volume_sma_20', 1)  # Default to 1 to avoid division by zero
        if volume_ma == 0:
            volume_ma = 1
            
        volume_ratio = volume / volume_ma
        
        if volume_ratio > 1.5:
            signals['volume'] = 'high'
        elif volume_ratio < 0.5:
            signals['volume'] = 'low'
        else:
            signals['volume'] = 'normal'
        
        # Price movement analysis
        if len(df) >= 24:
            price_24h_ago = df.iloc[-24]['close'] if len(df) >= 24 else df.iloc[0]['close']
            price_change_24h = (current['close'] - price_24h_ago) / price_24h_ago * 100
            
            if price_change_24h > 2:
                signals['price_24h'] = 'bullish'
            elif price_change_24h < -2:
                signals['price_24h'] = 'bearish'
            else:
                signals['price_24h'] = 'neutral'
        
        return signals
    
    def _add_ml_prediction(self, df: pd.DataFrame, signals: Dict[str, str]) -> Dict[str, str]:
        """Add machine learning-based prediction to signals."""
        # Only proceed if enough data is available
        if len(df) < 100:
            return signals
        
        try:
            # Feature selection for ML
            features = ['sma_7', 'sma_25', 'sma_99', 'rsi', 'macd', 'macd_hist', 
                        'bb_upper', 'bb_lower', 'stoch_k', 'stoch_d']
            
            # Keep only available features
            available_features = [f for f in features if f in df.columns]
            
            if len(available_features) < 5:  # Need at least some features
                return signals
            
            # Target: Was price higher in 24h?
            df['target'] = df['close'].shift(-24) > df['close']
            
            # Remove NaN values
            df_ml = df.dropna(subset=['target'] + available_features)
            
            if len(df_ml) < 50:
                return signals
                
            # Split into features and target
            X = df_ml[available_features].values
            y = df_ml['target'].values
            
            # Check if target has variation
            if len(np.unique(y)) < 2:
                return signals
                
            # Train a Random Forest model
            model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=5, 
                random_state=42, 
                n_jobs=-1  # Use all CPU cores
            )
            model.fit(X, y)
            
            # Get current features for prediction
            current_features = df[available_features].iloc[-1].values.reshape(1, -1)
            
            # Make prediction
            prediction_proba = model.predict_proba(current_features)[0]
            
            # Add prediction to signals based on probability
            if prediction_proba[1] > 0.65:
                signals['ml_prediction'] = 'bullish'
            elif prediction_proba[1] < 0.35:
                signals['ml_prediction'] = 'bearish'
            else:
                signals['ml_prediction'] = 'neutral'
            
            # Store confidence value
            signals['ml_confidence'] = float(max(prediction_proba))
                        # Add prediction to signals based on probability
            if prediction_proba[1] > 0.65:
                signals['ml_prediction'] = 'bullish'
            elif prediction_proba[1] < 0.35:
                signals['ml_prediction'] = 'bearish'
            else:
                signals['ml_prediction'] = 'neutral'
            
            # Store confidence value
            signals['ml_confidence'] = float(max(prediction_proba))
            
            return signals
            
        except Exception as e:
            logger.warning(f"Error in ML prediction: {e}")
            return signals
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate market volatility."""
        try:
            # Use the built-in volatility measure if available
            if 'volatility_14' in df.columns:
                volatility = df['volatility_14'].iloc[-1]
                
                # Normalize to 0-1 range
                return min(max(volatility * 10, 0), 1)
            
            # Calculate ATR-based volatility
            if 'atr_14' in df.columns and 'close' in df.columns:
                # ATR relative to price (normalized)
                recent_close = df['close'].iloc[-1]
                if recent_close > 0:
                    return min(df['atr_14'].iloc[-1] / recent_close * 100, 1)
            
            # Fallback: Calculate standard deviation based volatility
            if len(df) >= 14:
                # Get last 14 periods
                recent_data = df.iloc[-14:]
                # Calculate normalized volatility
                volatility = recent_data['close'].std() / recent_data['close'].mean()
                return min(volatility * 10, 1)
            
            # Default return if no calculation was possible
            return 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating volatility: {e}")
            return 0.5
    
    def _calculate_weighted_score(self, signals: Dict[str, str]) -> Tuple[float, float, float, str]:
        """Calculate weighted sentiment scores from signals."""
        # Get the weights for available signals
        available_weights = {k: v for k, v in self.sentiment_weights.items() if k in signals}
        
        if not available_weights:
            return 0.5, 0.5, 0.5, 'neutral'
            
        total_weight = sum(available_weights.values())
        
        # Initialize scores
        bullish_score = 0
        bearish_score = 0
        
        # Calculate scores based on signal values
        for signal_name, signal_value in signals.items():
            if signal_name in available_weights:
                weight = available_weights[signal_name]
                
                # Bullish signals
                if signal_value in ['bullish', 'oversold', 'expanding']:
                    bullish_score += weight
                # Bearish signals
                elif signal_value in ['bearish', 'overbought', 'contracting']:
                    bearish_score += weight
                # Neutral signals split the weight
                elif signal_value == 'neutral':
                    bullish_score += weight * 0.5
                    bearish_score += weight * 0.5
        
        # Normalize scores
        bullish_normalized = bullish_score / total_weight if total_weight > 0 else 0.5
        bearish_normalized = bearish_score / total_weight if total_weight > 0 else 0.5
        
        # Calculate overall sentiment strength and direction
        if bullish_normalized > bearish_normalized * 1.2:  # 20% threshold for decisive bullish
            overall = 'bullish'
            strength = bullish_normalized
        elif bearish_normalized > bullish_normalized * 1.2:  # 20% threshold for decisive bearish
            overall = 'bearish'
            strength = bearish_normalized
        else:
            overall = 'neutral'
            strength = 0.5
        
        return bullish_normalized, bearish_normalized, strength, overall


# =============================================================================
# LSTM PRICE PREDICTOR
# =============================================================================

class LSTMPricePredictor:
    """
    Deep learning model for price prediction using LSTM neural networks.
    Optimized for GPU acceleration with TensorFlow.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.lookback_period = 60
        self.feature_columns = None
        self.target_column = 'close'
        self.target_idx = None
        self.model_path = None
        self.train_history = None
        self.gpu_accelerator = GPUAccelerator()
        self.last_training_time = None
    
    def initialize(self, lookback_period: int = 60, model_path: str = None) -> bool:
        """Initialize the predictor with settings."""
        self.lookback_period = lookback_period
        
        # Initialize GPU
        self.gpu_accelerator.initialize()
        
        # Set model path
        if model_path:
            self.model_path = model_path
            # Try to load existing model
            if os.path.exists(model_path):
                return self.load_model(model_path)
        
        return True
    
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build LSTM model architecture optimized for RTX 3080."""
        # Get the most appropriate ML configuration
        gpu_params = self.gpu_accelerator.optimize_for_training()
        use_mixed_precision = gpu_params['use_mixed_precision']
        
        # Build the model
        model = tf.keras.Sequential()
        
        # Input LSTM layer
        model.add(tf.keras.layers.LSTM(
            units=ML_CONFIG['lstm_units'][0],
            return_sequences=True,
            input_shape=input_shape,
            kernel_initializer='he_normal',
            recurrent_activation='sigmoid'
        ))
        model.add(tf.keras.layers.Dropout(ML_CONFIG['dropout_rate']))
        
        # Middle LSTM layer
        model.add(tf.keras.layers.LSTM(
            units=ML_CONFIG['lstm_units'][1],
            return_sequences=True,
            kernel_initializer='he_normal',
            recurrent_activation='sigmoid'
        ))
        model.add(tf.keras.layers.Dropout(ML_CONFIG['dropout_rate']))
        
        # Final LSTM layer
        model.add(tf.keras.layers.LSTM(
            units=ML_CONFIG['lstm_units'][2],
            return_sequences=False,
            kernel_initializer='he_normal',
            recurrent_activation='sigmoid'
        ))
        model.add(tf.keras.layers.Dropout(ML_CONFIG['dropout_rate']))
        
        # Output layers
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(units=1))
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=ML_CONFIG['learning_rate'])
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        
        logger.info(f"LSTM model built successfully, input shape: {input_shape}")
        if self.gpu_accelerator.gpu_available:
            logger.info(f"Using GPU acceleration with mixed precision: {use_mixed_precision}")
        
        return model
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM model."""
        if df is None or len(df) < (self.lookback_period + 10):
            raise ValueError(f"Insufficient data points: {len(df) if df is not None else 0}")
        
        # Ensure DataFrame has no NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Select features - dynamically based on what's available
        essential_features = ['open', 'high', 'low', 'close', 'volume']
        
        # Add technical indicators if available
        potential_indicators = [
            'sma_7', 'sma_25', 'sma_99', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_hist', 'rsi',
            'bb_middle', 'bb_upper', 'bb_lower', 'stoch_k', 'stoch_d'
        ]
        
        # Start with essential features
        features = [f for f in essential_features if f in df.columns]
        
        # Add available indicators
        for indicator in potential_indicators:
            if indicator in df.columns:
                features.append(indicator)
        
        logger.info(f"Using {len(features)} features for LSTM model")
        self.feature_columns = features
        
        # Find target column index for later
        self.target_idx = features.index(self.target_column) if self.target_column in features else 0
        
        # Extract feature data
        dataset = df[features].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(dataset)
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback_period, len(scaled_data)):
            X.append(scaled_data[i-self.lookback_period:i])
            y.append(scaled_data[i, self.target_idx])
        
        return np.array(X), np.array(y)
    
    def train(self, df: pd.DataFrame, epochs: int = None, batch_size: int = None, 
              validation_split: float = 0.2, early_stopping: bool = True) -> Dict[str, Any]:
        """Train the LSTM model with the provided data."""
        try:
            # If no custom epochs/batch_size provided, use config values
            epochs = epochs or ML_CONFIG['epochs']
            batch_size = batch_size or ML_CONFIG['batch_size']
            
            logger.info(f"Preparing data for training from {len(df)} data points")
            X, y = self.prepare_data(df)
            
            # Split into training and validation sets
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
            
            # Build model if not already built
            if self.model is None:
                input_shape = (X_train.shape[1], X_train.shape[2])
                self.model = self.build_model(input_shape)
            
            # Setup callbacks
            callbacks = []
            
            if early_stopping:
                callbacks.append(tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=ML_CONFIG['early_stopping_patience'],
                    restore_best_weights=True,
                    verbose=1
                ))
            
            # Learning rate reduction
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=ML_CONFIG['reduce_lr_patience'],
                min_lr=0.0001,
                verbose=1
            ))
            
            # Create model directory if needed
            if self.model_path:
                model_dir = os.path.dirname(self.model_path)
                if model_dir and not os.path.exists(model_dir):
                    os.makedirs(model_dir, exist_ok=True)
                
                # Model checkpoint
                callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                ))
            
            # Train the model
            logger.info(f"Starting model training for {epochs} epochs with batch size {batch_size}")
            try:
                history = self.model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                    verbose=1
                )
                self.train_history = history.history
                
            except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
                logger.warning(f"GPU error during training: {e}")
                logger.info("Retrying with reduced batch size on CPU")
                
                # Fall back to CPU with smaller batch size
                with tf.device('/CPU:0'):
                    reduced_batch = max(batch_size // 2, 8)
                    history = self.model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=reduced_batch,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks,
                        verbose=1
                    )
                    self.train_history = history.history
            
            # Record training time
            self.last_training_time = datetime.datetime.now()
            
            # Evaluate model
            val_loss, val_mae = self.model.evaluate(X_val, y_val, verbose=0)
            logger.info(f"Training completed - Validation loss: {val_loss:.4f}, MAE: {val_mae:.4f}")
            
            # Save model if path provided
            if self.model_path and not any(isinstance(cb, tf.keras.callbacks.ModelCheckpoint) for cb in callbacks):
                self.model.save(self.model_path)
                logger.info(f"Model saved to {self.model_path}")
            
            # Return training results
            return {
                'val_loss': val_loss,
                'val_mae': val_mae,
                'history': self.train_history,
                'training_time': self.last_training_time
            }
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make price predictions using the trained model."""
        if self.model is None:
            logger.error("Model not trained or loaded")
            return None
        
        try:
            # Ensure we have enough data points
            if len(df) < self.lookback_period:
                logger.error(f"Not enough data points for prediction: {len(df)} < {self.lookback_period}")
                return None
            
            # Get feature data
            if not self.feature_columns:
                logger.warning("Feature columns not defined, using all available numeric columns")
                # Use all numeric columns
                features = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            else:
                # Use previously defined features if available
                features = [f for f in self.feature_columns if f in df.columns]
            
            # Check if we have the target column
            if self.target_column not in features:
                logger.warning(f"Target column {self.target_column} not in features, using first column")
                self.target_idx = 0
            else:
                self.target_idx = features.index(self.target_column)
            
            # Get the current price
            current_price = df[self.target_column].iloc[-1]
            
            # Extract the most recent data points
            recent_data = df[features].iloc[-self.lookback_period:].values
            
            # Scale the data
            if hasattr(self, 'scaler') and self.scaler is not None:
                try:
                    scaled_data = self.scaler.transform(recent_data)
                except (ValueError, AttributeError):
                    logger.warning("Error with existing scaler, creating new one")
                    self.scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = self.scaler.fit_transform(recent_data)
            else:
                self.scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = self.scaler.fit_transform(recent_data)
            
            # Reshape for LSTM [1, lookback_period, n_features]
            X_pred = np.array([scaled_data])
            
            # Make prediction (with reduced verbosity)
            predicted_scaled = self.model.predict(X_pred, verbose=0)
            
            # Reverse the scaling to get the actual predicted price
            dummy_array = np.zeros((1, len(features)))
            dummy_array[0, self.target_idx] = predicted_scaled[0, 0]
            predicted_price = self.scaler.inverse_transform(dummy_array)[0, self.target_idx]
            
            # Calculate price change
            price_change_pct = ((predicted_price - current_price) / current_price) * 100
            
            # Prepare prediction result
            result = {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change_percent': price_change_pct,
                'prediction_time': datetime.datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during price prediction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_model(self, model_path: str) -> bool:
        """Load a pre-trained model from file."""
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            self.model = tf.keras.models.load_model(model_path)
            self.model_path = model_path
            logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def save_model(self, model_path: str = None) -> bool:
        """Save the current model to file."""
        if self.model is None:
            logger.error("No model to save")
            return False
        
        try:
            path = model_path or self.model_path
            if not path:
                logger.error("No model path specified")
                return False
            
            # Create directory if it doesn't exist
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            self.model.save(path)
            logger.info(f"Model saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False


# =============================================================================
# EXCHANGE API MANAGER
# =============================================================================

class ExchangeAPIManager:
    """
    Manages connections and interactions with cryptocurrency exchanges.
    Handles rate limiting, error handling, and retry logic.
    """
    
    def __init__(self, exchange_id: str, api_key: str = None, api_secret: str = None, 
                 password: str = None, timeout: int = 30000):
        """Initialize the exchange connection."""
        self.exchange_id = exchange_id.lower()
        self.api_key = api_key
        self.api_secret = api_secret
        self.password = password
        self.timeout = timeout
        self.exchange = None
        self.markets = {}
        self.initialized = False
        self.has_trading_api = api_key is not None and api_secret is not None
        self.last_api_calls = {}
        self.api_call_count = {}
        
    def initialize(self) -> bool:
        """Initialize the exchange connection."""
        try:
            # Exchange configuration
            config = {
                'enableRateLimit': True,
                'timeout': self.timeout
            }
            
            # Add API credentials if available
            if self.api_key and self.api_secret:
                config['apiKey'] = self.api_key
                config['secret'] = self.api_secret
                
                # Some exchanges require additional parameters
                if self.password and self.exchange_id == 'kucoin':
                    config['password'] = self.password
            
            # Create the exchange instance
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class(config)
            
            # Load markets (with retry)
            self.markets = self._call_with_retry(self.exchange.load_markets)
            
            self.initialized = True
            logger.info(f"Exchange {self.exchange_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing exchange {self.exchange_id}: {e}")
            self.initialized = False
            return False
    
    def _call_with_retry(self, method, *args, max_retries=EXCHANGE_CONFIG['retry_attempts'], **kwargs):
        """Call an exchange API method with retry logic."""
        if not self.initialized and method != self.exchange.load_markets:
            if not self.initialize():
                raise Exception(f"Exchange {self.exchange_id} is not initialized")
        
        # Track API calls
        method_name = method.__name__ if hasattr(method, '__name__') else str(method)
        self.last_api_calls[method_name] = datetime.datetime.now()
        self.api_call_count[method_name] = self.api_call_count.get(method_name, 0) + 1
        
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                return method(*args, **kwargs)
            
            except ccxt.DDoSProtection as e:
                # Rate limit error, wait with exponential backoff
                retry_count += 1
                wait_time = EXCHANGE_CONFIG['backoff_factor'] ** retry_count
                
                logger.warning(f"Rate limit hit for {self.exchange_id}, waiting {wait_time}s before retry {retry_count}/{max_retries}")
                time.sleep(wait_time)
                last_error = e
            
            except ccxt.NetworkError as e:
                # Network error, retry with backoff
                retry_count += 1
                wait_time = EXCHANGE_CONFIG['backoff_factor'] ** retry_count
                
                logger.warning(f"Network error for {self.exchange_id}, waiting {wait_time}s before retry {retry_count}/{max_retries}")
                time.sleep(wait_time)
                last_error = e
            
            except ccxt.ExchangeError as e:
                # Exchange error, some can be retried
                if any(err in str(e).lower() for err in ['timeout', 'temporarily', 'overloaded']):
                    retry_count += 1
                    wait_time = EXCHANGE_CONFIG['backoff_factor'] ** retry_count
                    
                    logger.warning(f"Exchange error for {self.exchange_id}, waiting {wait_time}s before retry {retry_count}/{max_retries}")
                    time.sleep(wait_time)
                    last_error = e
                else:
                    # Non-retryable exchange error
                    logger.error(f"Exchange error for {self.exchange_id}: {e}")
                    raise
            
            except Exception as e:
                # Other errors, don't retry
                logger.error(f"Unexpected error with {self.exchange_id}: {e}")
                raise
        
        # If we've exhausted retries, raise the last error
        if last_error:
            logger.error(f"Max retries exceeded for {self.exchange_id}: {last_error}")
            raise last_error
        
        return None
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data with error handling and conversion to DataFrame."""
        try:
            # Check if timeframe is supported
            if not self.initialized:
                self.initialize()
                
            if hasattr(self.exchange, 'timeframes') and timeframe not in self.exchange.timeframes:
                supported = list(self.exchange.timeframes.keys())
                logger.error(f"Timeframe {timeframe} not supported by {self.exchange_id}. Supported: {supported}")
                return None
            
            # Fetch OHLCV data
            ohlcv = self._call_with_retry(self.exchange.fetch_ohlcv, symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) == 0:
                logger.warning(f"No OHLCV data returned for {symbol} on {self.exchange_id}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Validate data
            self._validate_ohlcv_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol} on {self.exchange_id}: {e}")
            return None
    
    def _validate_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean OHLCV data."""
        if df is None or len(df) == 0:
            return df
        
        # Check for NaN values
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in OHLCV data")
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Check for negative values in price
        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            logger.warning("Found non-positive values in price data")
            # Replace non-positive values with small positive number
            for col in ['open', 'high', 'low', 'close']:
                df.loc[df[col] <= 0, col] = 0.00000001
        
        # Check for high-low inconsistency
        if (df['high'] < df['low']).any():
            logger.warning("Found inconsistency: high < low")
            # Swap high and low where needed
            invalid_rows = df['high'] < df['low']
            temp = df.loc[invalid_rows, 'high'].copy()
            df.loc[invalid_rows, 'high'] = df.loc[invalid_rows, 'low']
            df.loc[invalid_rows, 'low'] = temp
        
        # Check for other inconsistencies
        if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
            logger.warning("Found inconsistency: high < open/close")
            # Fix highs
            df['high'] = df[['high', 'open', 'close']].max(axis=1)
        
        if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
            logger.warning("Found inconsistency: low > open/close")
            # Fix lows
            df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        return df
    
    def fetch_balance(self) -> Dict[str, Any]:
        """Fetch account balance with error handling."""
        if not self.has_trading_api:
            logger.warning("Cannot fetch balance without API credentials")
            return None
        
        try:
            balance = self._call_with_retry(self.exchange.fetch_balance)
            
            # Simplify the balance structure
            result = {
                'total': balance.get('total', {}),
                'free': balance.get('free', {}),
                'used': balance.get('used', {})
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching balance on {self.exchange_id}: {e}")
            return None
    
    def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: float = None) -> Dict[str, Any]:
        """Create a new order with error handling and validation."""
        if not self.has_trading_api:
            logger.warning("Cannot create order without API credentials")
            return None
        
        try:
            # Validate and format the order parameters
            order_params = self._validate_order_params(symbol, order_type, side, amount, price)
            if not order_params:
                return None
            
            # Extract validated parameters
            symbol = order_params['symbol']
            order_type = order_params['order_type']
            side = order_params['side']
            amount = order_params['amount']
            price = order_params.get('price')
            
            # Create the order
            if order_type.lower() == 'market':
                order = self._call_with_retry(self.exchange.create_market_order, symbol, side, amount)
            else:
                if price is None:
                    raise ValueError("Price required for limit orders")
                order = self._call_with_retry(self.exchange.create_limit_order, symbol, side, amount, price)
            
            logger.info(f"Order created: {side.upper()} {amount} {symbol} at {price if price else 'market price'}")
            return order
            
        except Exception as e:
            logger.error(f"Error creating order on {self.exchange_id}: {e}")
            return None
    
    def _validate_order_params(self, symbol: str, order_type: str, side: str, amount: float, price: float = None) -> Dict[str, Any]:
        """Validate and format order parameters."""
        try:
            # Ensure the exchange is initialized
            if not self.initialized:
                if not self.initialize():
                    return None
            
            # Validate symbol
            if symbol not in self.markets:
                logger.error(f"Symbol {symbol} not found in available markets")
                return None
            
            # Validate order type
            order_type = order_type.lower()
            if order_type not in ['market', 'limit']:
                logger.error(f"Order type {order_type} not supported, use 'market' or 'limit'")
                return None
            
            # Validate side
            side = side.lower()
            if side not in ['buy', 'sell']:
                logger.error(f"Side {side} not supported, use 'buy' or 'sell'")
                return None
            
            # Validate amount
            market_info = self.markets[symbol]
            
            # Validate amount
            market_info = self.markets[symbol]
            
            # Format the amount to proper precision
            if 'precision' in market_info and 'amount' in market_info['precision']:
                precision = market_info['precision']['amount']
                # If precision is an integer, it's decimal places
                if isinstance(precision, int):
                    amount = float(round(amount, precision))
                # If precision is a float, it's the lot size
                elif isinstance(precision, float):
                    amount = float(round(amount / precision) * precision)
            
            # Check minimum amount
            if 'limits' in market_info and 'amount' in market_info['limits']:
                min_amount = market_info['limits']['amount']['min']
                if min_amount and amount < min_amount:
                    logger.error(f"Amount {amount} for {symbol} is below minimum {min_amount}")
                    return None
                
                max_amount = market_info['limits']['amount']['max']
                if max_amount and amount > max_amount:
                    logger.error(f"Amount {amount} for {symbol} is above maximum {max_amount}")
                    return None
            
            # Validate price for limit orders
            if order_type == 'limit' and price is not None:
                # Format the price to proper precision
                if 'precision' in market_info and 'price' in market_info['precision']:
                    precision = market_info['precision']['price']
                    if isinstance(precision, int):
                        price = float(round(price, precision))
                    elif isinstance(precision, float):
                        price = float(round(price / precision) * precision)
                
                # Check minimum price
                if 'limits' in market_info and 'price' in market_info['limits']:
                    min_price = market_info['limits']['price']['min']
                    if min_price and price < min_price:
                        logger.error(f"Price {price} for {symbol} is below minimum {min_price}")
                        return None
                    
                    max_price = market_info['limits']['price']['max']
                    if max_price and price > max_price:
                        logger.error(f"Price {price} for {symbol} is above maximum {max_price}")
                        return None
            
            # Check order cost (amount * price)
            order_cost = amount * (price if price else 0)
            if 'limits' in market_info and 'cost' in market_info['limits']:
                min_cost = market_info['limits']['cost']['min']
                if min_cost and order_cost < min_cost:
                    logger.error(f"Order cost {order_cost} for {symbol} is below minimum {min_cost}")
                    return None
            
            # Return validated parameters
            result = {
                'symbol': symbol,
                'order_type': order_type,
                'side': side,
                'amount': amount
            }
            
            if order_type == 'limit' and price is not None:
                result['price'] = price
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating order parameters: {e}")
            return None
    
    def fetch_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Fetch the status of an order."""
        if not self.has_trading_api:
            logger.warning("Cannot fetch order status without API credentials")
            return None
        
        try:
            order = self._call_with_retry(self.exchange.fetch_order, order_id, symbol)
            return order
        except Exception as e:
            logger.error(f"Error fetching order status for {order_id} on {self.exchange_id}: {e}")
            return None
    
    def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """Get market information for a symbol."""
        try:
            if not self.initialized:
                self.initialize()
                
            if symbol not in self.markets:
                logger.warning(f"Symbol {symbol} not found in available markets")
                return None
                
            return self.markets[symbol]
        except Exception as e:
            logger.error(f"Error getting market info for {symbol}: {e}")
            return None
    
    def get_min_notional(self, symbol: str) -> float:
        """Get minimum notional value for a symbol."""
        market_info = self.get_market_info(symbol)
        if not market_info:
            return EXCHANGE_CONFIG['default_min_notional']
        
        try:
            if 'limits' in market_info and 'cost' in market_info['limits'] and 'min' in market_info['limits']['cost']:
                return market_info['limits']['cost']['min']
        except Exception:
            pass
        
        return EXCHANGE_CONFIG['default_min_notional']
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        try:
            ticker = self._call_with_retry(self.exchange.fetch_ticker, symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None


# =============================================================================
# CRYPTO TRADING BOT
# =============================================================================

class CryptoTradingBot:
    """
    Main trading bot class that integrates all components.
    Handles trading decisions, execution, and performance tracking.
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, 
                 symbol: str = 'BTC/USDT', timeframe: str = '1h', 
                 exchange: str = 'binance', base_dir: str = None):
        """Initialize the trading bot with configuration."""
        # Trading parameters
        self.symbol = symbol
        self.timeframe = timeframe
        self.trade_amount = TRADING_PARAMS['default_trade_amount']
        self.stop_loss = TRADING_PARAMS['default_stop_loss']
        self.take_profit = TRADING_PARAMS['default_take_profit']
        
        # API credentials
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Initialize exchange connection
        self.exchange_manager = ExchangeAPIManager(exchange, api_key, api_secret)
        self.exchange_manager.initialize()
        
        # Initialize components
        self.price_predictor = LSTMPricePredictor()
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        self.gpu_accelerator = GPUAccelerator()
        self.gpu_accelerator.initialize()
        
        # Set up directories
        if base_dir:
            self.base_dir = base_dir
        else:
            self.base_dir = os.path.join(os.getcwd(), 'trading_bot_data')
        
        self.model_dir = os.path.join(self.base_dir, 'models')
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.visualization_dir = os.path.join(self.base_dir, 'visualizations')
        
        for directory in [self.base_dir, self.model_dir, self.data_dir, self.log_dir, self.visualization_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Setup model path
        symbol_id = symbol.replace('/', '_')
        self.model_path = os.path.join(self.model_dir, f"{symbol_id}_{timeframe}_model.h5")
        
        # Initialize price predictor with model path
        self.price_predictor.initialize(model_path=self.model_path)
        
        # Performance tracking
        self.trades_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'start_balance': 0.0,
            'current_balance': 0.0
        }
        
        # Runtime state
        self.is_running = False
        self.stop_requested = False
        self.last_analysis_time = None
        self.last_training_time = None
        self.api_calls = {}
        self.demo_mode = api_key is None or api_key == "DEMO_MODE"
        
        logger.info(f"Trading bot initialized for {symbol} on {exchange}")
        if self.demo_mode:
            logger.info("Running in DEMO MODE - no real trades will be executed")
    
    def fetch_historical_data(self, limit: int = 500) -> pd.DataFrame:
        """Fetch historical market data with technical indicators."""
        df = self.exchange_manager.fetch_ohlcv(self.symbol, self.timeframe, limit)
        
        if df is not None and len(df) > 0:
            # Add technical indicators
            df = TechnicalIndicators.add_all_indicators(df)
            logger.info(f"Fetched {len(df)} data points with technical indicators")
            return df
        else:
            logger.error("Failed to fetch historical data")
            return None
    
    def train_model(self, force: bool = False) -> Dict[str, Any]:
        """Train or retrain the prediction model."""
        # Check if training is necessary
        if not force and self.price_predictor.model is not None and self.price_predictor.last_training_time is not None:
            time_since_last_training = datetime.datetime.now() - self.price_predictor.last_training_time
            if time_since_last_training.total_seconds() < 12 * 3600:  # 12 hours
                logger.info(f"Skipping training - last training was {time_since_last_training.total_seconds() / 3600:.1f} hours ago")
                return {"status": "skipped", "reason": "recent_training"}
        
        # Fetch more data for training
        df = self.fetch_historical_data(limit=2000)
        
        if df is None or len(df) < 200:
            logger.error("Insufficient data for model training")
            return {"status": "error", "reason": "insufficient_data"}
        
        # Train the model
        logger.info("Starting model training")
        result = self.price_predictor.train(df)
        
        if "error" in result:
            logger.error(f"Training failed: {result['error']}")
            return {"status": "error", "reason": result["error"]}
        
        self.last_training_time = datetime.datetime.now()
        logger.info(f"Model training completed - Validation MAE: {result.get('val_mae', 'N/A')}")
        
        return {"status": "success", "result": result}
    
    def predict_next_price(self) -> Dict[str, Any]:
        """Predict the next price movement."""
        # Fetch recent data
        df = self.fetch_historical_data(limit=self.price_predictor.lookback_period + 20)
        
        if df is None or len(df) < self.price_predictor.lookback_period:
            logger.error("Insufficient data for price prediction")
            return None
        
        # Make prediction
        prediction = self.price_predictor.predict(df)
        
        if prediction:
            logger.info(f"Predicted price: {prediction['predicted_price']:.2f} (Change: {prediction['price_change_percent']:.2f}%)")
        else:
            logger.error("Price prediction failed")
        
        return prediction
    
    def get_market_sentiment(self) -> Dict[str, Any]:
        """Analyze market sentiment."""
        # Fetch data for sentiment analysis
        df = self.fetch_historical_data(limit=200)
        
        if df is None or len(df) < 50:
            logger.error("Insufficient data for sentiment analysis")
            return None
        
        # Analyze sentiment
        sentiment = self.sentiment_analyzer.analyze(df)
        
        if sentiment:
            logger.info(f"Market sentiment: {sentiment['overall'].upper()} (Strength: {sentiment['strength']:.2f})")
        else:
            logger.error("Sentiment analysis failed")
        
        return sentiment
    
    def check_balance(self) -> Dict[str, Any]:
        """Check account balance."""
        balance = self.exchange_manager.fetch_balance()
        
        if balance:
            base_currency, quote_currency = self.symbol.split('/')
            base_amount = balance['total'].get(base_currency, 0)
            quote_amount = balance['total'].get(quote_currency, 0)
            
            logger.info(f"Balance: {base_amount} {base_currency}, {quote_amount} {quote_currency}")
            
            # Update initial balance if not set
            if self.performance_metrics['start_balance'] == 0:
                self.performance_metrics['start_balance'] = quote_amount
                self.performance_metrics['current_balance'] = quote_amount
            else:
                self.performance_metrics['current_balance'] = quote_amount
            
            # Enhanced result with additional info
            result = {
                'total': balance['total'],
                'free': balance.get('free', {}),
                'used': balance.get('used', {}),
                base_currency: base_amount,
                quote_currency: quote_amount
            }
            
            return result
        else:
            if self.demo_mode:
                # Provide mock balance in demo mode
                base_currency, quote_currency = self.symbol.split('/')
                mock_balance = {
                    'total': {base_currency: 1.0, quote_currency: 10000.0},
                    'free': {base_currency: 1.0, quote_currency: 10000.0},
                    base_currency: 1.0,
                    quote_currency: 10000.0
                }
                logger.info(f"DEMO MODE - Mock balance: 1.0 {base_currency}, 10000.0 {quote_currency}")
                return mock_balance
            else:
                logger.error("Failed to fetch balance")
                return None
    
    def execute_strategy(self) -> Dict[str, Any]:
        """Execute the trading strategy based on analysis."""
        result = {
            'trade_executed': False,
            'action': None,
            'amount': 0,
            'price': 0,
            'reason': None
        }
        
        try:
            # Record analysis time
            self.last_analysis_time = datetime.datetime.now()
            
            # Check market status
            market_status = self._check_market_status()
            if market_status and market_status.get('should_pause', False):
                reason = market_status.get('reason', 'Unknown reason')
                logger.warning(f"Trading paused: {reason}")
                result['reason'] = f"Trading paused: {reason}"
                return result
            
            # Get account balance
            balance_info = self.check_balance()
            if not balance_info:
                logger.error("Failed to get account balance")
                result['reason'] = "Failed to get account balance"
                return result
            
            # Get market sentiment and price prediction
            sentiment = self.get_market_sentiment()
            prediction = self.predict_next_price()
            
            # Validate results
            if not sentiment or not prediction:
                missing = []
                if not sentiment:
                    missing.append("sentiment analysis")
                if not prediction:
                    missing.append("price prediction")
                
                reason = f"Missing data: {', '.join(missing)}"
                logger.warning(reason)
                result['reason'] = reason
                return result
            
            # Parse symbol
            base_currency, quote_currency = self.symbol.split('/')
            
            # Get available balances
            base_balance = balance_info.get(base_currency, 0)
            quote_balance = balance_info.get(quote_currency, 0)
            
            # Use free balance if available
            if 'free' in balance_info:
                base_free = balance_info['free'].get(base_currency, 0)
                quote_free = balance_info['free'].get(quote_currency, 0)
                
                if base_free < base_balance:
                    logger.info(f"Using free {base_currency} balance: {base_free} of {base_balance}")
                    base_balance = base_free
                
                if quote_free < quote_balance:
                    logger.info(f"Using free {quote_currency} balance: {quote_free} of {quote_balance}")
                    quote_balance = quote_free
            
            # Current market price
            current_price = prediction['current_price']
            predicted_price = prediction['predicted_price']
            price_change_pct = prediction['price_change_percent']
            
            # Log analysis details
            logger.info(f"Price: {current_price} {quote_currency} | Prediction: {predicted_price} {quote_currency} ({price_change_pct:+.2f}%)")
            logger.info(f"Sentiment: {sentiment['overall'].upper()} (Strength: {sentiment['strength']:.2f})")
            
            # Adjust trading size based on volatility
            adjusted_trade_amount = self.trade_amount
            if 'volatility' in sentiment:
                volatility = sentiment['volatility']
                if volatility > TRADING_PARAMS['high_volatility_threshold']:
                    adjusted_trade_amount *= 0.5
                    logger.info(f"High volatility ({volatility:.2f}) - Reducing trade size by 50%")
                elif volatility > TRADING_PARAMS['med_volatility_threshold']:
                    adjusted_trade_amount *= 0.75
                    logger.info(f"Medium volatility ({volatility:.2f}) - Reducing trade size by 25%")
            
            # Determine trading signal
            signal, reason = self._determine_trading_signal(sentiment, price_change_pct)
            logger.info(f"Trading signal: {signal} - {reason}")
            
            # Reserve minimum quote amount
            min_quote_reserve = TRADING_PARAMS['min_quote_reserve']
            available_quote = max(0, quote_balance - min_quote_reserve)
            
            # Execute trade based on signal
            if signal in ['strong_buy', 'buy'] and available_quote > 0:
                # Calculate buy size
                trade_ratio = adjusted_trade_amount
                if signal == 'strong_buy':
                    trade_ratio = min(adjusted_trade_amount * 1.5, 0.25)
                
                amount_to_buy = available_quote * trade_ratio / current_price
                
                # Check minimum notional value
                min_notional = self.exchange_manager.get_min_notional(self.symbol)
                if amount_to_buy * current_price < min_notional:
                    if available_quote >= min_notional:
                        amount_to_buy = min_notional / current_price * 1.01
                        logger.info(f"Adjusted buy amount to meet minimum notional: {amount_to_buy} {base_currency}")
                    else:
                        logger.warning(f"Insufficient funds for minimum notional: {available_quote} < {min_notional} {quote_currency}")
                        result['reason'] = f"Insufficient funds for minimum notional: {available_quote} < {min_notional} {quote_currency}"
                        return result
                
                # Set limit price slightly above market for quicker execution
                limit_price = current_price * 1.001
                
                # Execute or simulate order
                order = None
                if self.demo_mode:
                    logger.info(f"DEMO MODE - Buy Order: {amount_to_buy} {base_currency} @ {limit_price} {quote_currency}")
                    order = {
                        "id": f"demo_{datetime.datetime.now().timestamp()}",
                        "status": "closed",
                        "filled": amount_to_buy,
                        "price": limit_price,
                        "cost": amount_to_buy * limit_price
                    }
                else:
                    order = self.exchange_manager.create_order(
                        self.symbol, 'limit', 'buy', amount_to_buy, limit_price
                    )
                
                if order:
                    logger.info(f"Buy order executed: {amount_to_buy} {base_currency} @ {limit_price} {quote_currency}")
                    
                    # Update statistics
                    self.performance_metrics['total_trades'] += 1
                    
                    # Record trade
                    trade_record = {
                        'timestamp': datetime.datetime.now().isoformat(),
                        'action': 'BUY',
                        'amount': amount_to_buy,
                        'price': limit_price,
                        'value': amount_to_buy * limit_price,
                        'reason': reason,
                        'order_id': order.get('id', 'unknown')
                    }
                    self.trades_history.append(trade_record)
                    
                    # Update result
                    result['trade_executed'] = True
                    result['action'] = 'BUY'
                    result['amount'] = amount_to_buy
                    result['price'] = limit_price
                    result['reason'] = reason
                    result['order'] = order
                    
                    return result
                else:
                    logger.error("Failed to create buy order")
                    result['reason'] = "Failed to create buy order"
                    return result
                
            elif signal in ['strong_sell', 'sell'] and base_balance > 0:
                # Calculate sell size
                trade_ratio = adjusted_trade_amount
                if signal == 'strong_sell':
                    trade_ratio = min(adjusted_trade_amount * 1.5, 0.25)
                
                amount_to_sell = base_balance * trade_ratio
                
                # Check minimum notional
                min_notional = self.exchange_manager.get_min_notional(self.symbol)
                if amount_to_sell * current_price < min_notional:
                    if base_balance * current_price >= min_notional:
                        amount_to_sell = min_notional / current_price * 1.01
                        logger.info(f"Adjusted sell amount to meet minimum notional: {amount_to_sell} {base_currency}")
                    else:
                        logger.warning(f"Insufficient {base_currency} for minimum notional: {base_balance * current_price} < {min_notional} {quote_currency}")
                        result['reason'] = f"Insufficient {base_currency} for minimum notional"
                        return result
                
                # Set limit price slightly below market for quicker execution
                limit_price = current_price * 0.999
                
                # Execute or simulate order
                order = None
                if self.demo_mode:
                    logger.info(f"DEMO MODE - Sell Order: {amount_to_sell} {base_currency} @ {limit_price} {quote_currency}")
                    order = {
                        "id": f"demo_{datetime.datetime.now().timestamp()}",
                        "status": "closed",
                        "filled": amount_to_sell,
                        "price": limit_price,
                        "cost": amount_to_sell * limit_price
                    }
                else:
                    order = self.exchange_manager.create_order(
                        self.symbol, 'limit', 'sell', amount_to_sell, limit_price
                    )
                
                if order:
                    logger.info(f"Sell order executed: {amount_to_sell} {base_currency} @ {limit_price} {quote_currency}")
                    
                    # Update statistics
                    self.performance_metrics['total_trades'] += 1
                    
                    # Record trade
                    trade_record = {
                        'timestamp': datetime.datetime.now().isoformat(),
                        'action': 'SELL',
                        'amount': amount_to_sell,
                        'price': limit_price,
                        'value': amount_to_sell * limit_price,
                        'reason': reason,
                        'order_id': order.get('id', 'unknown')
                    }
                    self.trades_history.append(trade_record)
                    
                    # Update result
                    result['trade_executed'] = True
                    result['action'] = 'SELL'
                    result['amount'] = amount_to_sell
                    result['price'] = limit_price
                    result['reason'] = reason
                    result['order'] = order
                    
                    return result
                else:
                    logger.error("Failed to create sell order")
                    result['reason'] = "Failed to create sell order"
                    return result
            
            else:
                logger.info(f"No trade executed - {signal.upper()}: {reason}")
                result['reason'] = reason
                return result
            
        except Exception as e:
            logger.error(f"Error executing strategy: {e}")
            import traceback
            traceback.print_exc()
            result['reason'] = f"Error: {str(e)}"
            return result
    
    def _determine_trading_signal(self, sentiment: Dict[str, Any], price_change_pct: float) -> Tuple[str, str]:
        """Determine trading signal based on sentiment and price prediction."""
        # Strong buy signal
        if sentiment['overall'] == 'bullish' and price_change_pct > 1.5:
            return 'strong_buy', f"Strong buy signal: Bullish sentiment ({sentiment['strength']:.2f}) + high price increase predicted (+{price_change_pct:.2f}%)"
        
        # Strong sell signal
        elif sentiment['overall'] == 'bearish' and price_change_pct < -1.5:
            return 'strong_sell', f"Strong sell signal: Bearish sentiment ({sentiment['strength']:.2f}) + high price decrease predicted ({price_change_pct:.2f}%)"
        
        # Buy signal
        elif sentiment['overall'] == 'bullish' and price_change_pct > 0.5:
            return 'buy', f"Buy signal: Bullish sentiment ({sentiment['strength']:.2f}) + moderate price increase predicted (+{price_change_pct:.2f}%)"
        
        # Sell signal
        elif sentiment['overall'] == 'bearish' and price_change_pct < -0.5:
            return 'sell', f"Sell signal: Bearish sentiment ({sentiment['strength']:.2f}) + moderate price decrease predicted ({price_change_pct:.2f}%)"
        
        # Neutral market
        elif sentiment['overall'] == 'neutral' and abs(price_change_pct) < 0.5:
            return 'hold', f"Hold signal: Neutral sentiment with small predicted price change ({price_change_pct:.2f}%)"
        
        # Conflicting signals but strong sentiment
        elif sentiment['strength'] > 0.7:
            if sentiment['overall'] == 'bullish':
                return 'buy', f"Buy signal: Strong bullish sentiment ({sentiment['strength']:.2f}) despite conflicting price prediction"
            else:
                return 'sell', f"Sell signal: Strong bearish sentiment ({sentiment['strength']:.2f}) despite conflicting price prediction"
        
        # Default to hold
        else:
            return 'hold', "Hold signal: Conflicting indicators with low confidence"
    
    def _check_market_status(self) -> Dict[str, Any]:
        """Check if market conditions are suitable for trading."""
        try:
            # Get current ticker data
            ticker = self.exchange_manager._call_with_retry(
                self.exchange_manager.exchange.fetch_ticker, self.symbol
            )
            
            if not ticker:
                return {'should_pause': True, 'reason': "Could not fetch market data"}
            
            # Check for extreme volatility
            if 'percentage' in ticker and abs(ticker['percentage']) > 10:
                return {
                    'should_pause': True,
                    'reason': f"Extreme volatility detected: {ticker['percentage']}% change"
                }
            
            # Check for abnormal trading volume
            if hasattr(self, 'average_volume') and self.average_volume > 0:
                if ticker.get('quoteVolume', 0) > self.average_volume * 5:
                    return {
                        'should_pause': True,
                        'reason': f"Abnormal trading volume: {ticker['quoteVolume']} (5x average)"
                    }
            
            # Check for market hours (if applicable)
            # Some markets might have maintenance windows or reduced liquidity periods
            
            # All checks passed
            return {'should_pause': False}
            
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            # Default to allowing trading in case of errors
            return {'should_pause': False}
    
    def _generate_performance_report(self, final: bool = False) -> Dict[str, Any]:
        """Generate a performance report with trade statistics."""
        try:
            # Get current balance
            balance_info = self.check_balance()
            
            if not balance_info:
                logger.warning("Could not generate performance report - balance info unavailable")
                return None
            
            # Extract quote currency
            _, quote_currency = self.symbol.split('/')
            
            # Get current balance
            current_balance = balance_info.get(quote_currency, 0)
            self.performance_metrics['current_balance'] = current_balance
            
            # Calculate overall profit/loss
            start_balance = self.performance_metrics['start_balance']
            total_profit = current_balance - start_balance
            total_profit_pct = 0
            
            if start_balance > 0:
                total_profit_pct = (total_profit / start_balance) * 100
            
            # Update profit metrics
            self.performance_metrics['total_profit'] = total_profit
            
            # Calculate max drawdown if we have trade history
            if len(self.trades_history) > 1:
                self._calculate_drawdown()
            
            # Prepare report
            report = {
                'timestamp': datetime.datetime.now().isoformat(),
                'symbol': self.symbol,
                'start_balance': start_balance,
                'current_balance': current_balance,
                'total_profit': total_profit,
                'total_profit_pct': total_profit_pct,
                'total_trades': self.performance_metrics['total_trades'],
                'winning_trades': self.performance_metrics['winning_trades'],
                'losing_trades': self.performance_metrics['losing_trades'],
                'max_drawdown': self.performance_metrics['max_drawdown']
            }
            
            # Calculate win rate
            if self.performance_metrics['total_trades'] > 0:
                report['win_rate'] = (self.performance_metrics['winning_trades'] / 
                                      self.performance_metrics['total_trades']) * 100
            else:
                report['win_rate'] = 0
            
            # Log report
            title = "=== FINAL PERFORMANCE REPORT ===" if final else "=== PERFORMANCE REPORT ==="
            logger.info(title)
            logger.info(f"Symbol: {self.symbol} | Timeframe: {self.timeframe}")
            logger.info(f"Start Balance: {start_balance:.2f} {quote_currency}")
            logger.info(f"Current Balance: {current_balance:.2f} {quote_
                        logger.info(f"Current Balance: {current_balance:.2f} {quote_currency}")
            logger.info(f"Total Profit: {total_profit:.2f} {quote_currency} ({total_profit_pct:.2f}%)")
            logger.info(f"Total Trades: {self.performance_metrics['total_trades']}")
            
            if self.performance_metrics['total_trades'] > 0:
                win_rate = report['win_rate']
                logger.info(f"Win Rate: {win_rate:.2f}%")
            
            logger.info(f"Max Drawdown: {self.performance_metrics['max_drawdown']:.2f}%")
            logger.info("=" * 50)
            
            # Save report to file
            self._save_performance_report(report, final)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return None
    
    def _calculate_drawdown(self):
        """Calculate maximum drawdown from trade history."""
        try:
            # Need at least two trades to calculate drawdown
            if len(self.trades_history) < 2:
                return
            
            # Extract quote currency
            _, quote_currency = self.symbol.split('/')
            
            # Start with initial balance
            balance = self.performance_metrics['start_balance']
            peak_balance = balance
            drawdown = 0
            max_drawdown = 0
            
            # Process each trade
            for trade in self.trades_history:
                action = trade['action']
                value = trade['value']
                
                # Update balance (simplified - ignores fees)
                if action == 'BUY':
                    balance -= value
                elif action == 'SELL':
                    balance += value
                
                # Update peak and drawdown
                if balance > peak_balance:
                    peak_balance = balance
                
                if peak_balance > 0:
                    drawdown = (peak_balance - balance) / peak_balance * 100
                    max_drawdown = max(max_drawdown, drawdown)
            
            # Update metric
            self.performance_metrics['max_drawdown'] = max_drawdown
            logger.info(f"Maximum drawdown calculated: {max_drawdown:.2f}%")
            
        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
    
    def _save_performance_report(self, report: Dict[str, Any], final: bool = False):
        """Save performance report to file."""
        try:
            # Create filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_{'final_' if final else ''}{self.symbol.replace('/', '_')}_{timestamp}.json"
            filepath = os.path.join(self.data_dir, filename)
            
            # Save as JSON
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Performance report saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving performance report: {e}")
    
    def export_trading_history(self, format: str = 'json'):
        """Export trading history in various formats."""
        try:
            if not self.trades_history:
                logger.warning("No trades to export")
                return None
            
            # Create filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"trades_{self.symbol.replace('/', '_')}_{timestamp}"
            
            if format.lower() == 'json':
                # Export as JSON
                filepath = os.path.join(self.data_dir, f"{base_filename}.json")
                with open(filepath, 'w') as f:
                    json.dump(self.trades_history, f, indent=2)
                
                logger.info(f"Trading history exported to {filepath}")
                return filepath
                
            elif format.lower() == 'csv':
                # Export as CSV
                filepath = os.path.join(self.data_dir, f"{base_filename}.csv")
                
                # Convert to DataFrame for easier CSV export
                import pandas as pd
                df = pd.DataFrame(self.trades_history)
                df.to_csv(filepath, index=False)
                
                logger.info(f"Trading history exported to {filepath}")
                return filepath
                
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
                
        except Exception as e:
            logger.error(f"Error exporting trading history: {e}")
            return None
    
    def run(self, interval_seconds: int = 60, training_interval_hours: int = 12):
        """Run the trading bot continuously."""
        self.is_running = True
        self.stop_requested = False
        
        logger.info(f"Starting trading bot for {self.symbol} with {interval_seconds}s interval")
        logger.info(f"Model will be trained every {training_interval_hours} hours")
        
        # Initial market data fetch to establish baseline
        try:
            df = self.fetch_historical_data(limit=500)
            if df is not None and len(df) > 100:
                volume_data = df['volume'].iloc[-30:].mean()
                self.average_volume = volume_data
                logger.info(f"Established baseline volume: {self.average_volume}")
            else:
                logger.warning("Could not establish baseline market data")
        except Exception as e:
            logger.error(f"Error establishing baseline: {e}")
        
        # Initial balance check
        self.check_balance()
        
        # Initial model training if needed
        if self.price_predictor.model is None:
            logger.info("Initial model training...")
            self.train_model()
        
        last_training_time = datetime.datetime.now()
        
        # Main loop
        try:
            while not self.stop_requested:
                loop_start_time = datetime.datetime.now()
                
                try:
                    # Check if model training is needed
                    time_since_training = (loop_start_time - last_training_time).total_seconds() / 3600
                    if time_since_training > training_interval_hours:
                        logger.info(f"Scheduled model training ({time_since_training:.1f} hours since last training)")
                        training_result = self.train_model()
                        if training_result.get('status') == 'success':
                            last_training_time = datetime.datetime.now()
                    
                    # Execute the trading strategy
                    logger.info("Executing trading strategy...")
                    strategy_result = self.execute_strategy()
                    
                    # Generate performance report if trade executed
                    if strategy_result and strategy_result.get('trade_executed', False):
                        self._generate_performance_report()
                    
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Calculate time to sleep
                elapsed = (datetime.datetime.now() - loop_start_time).total_seconds()
                sleep_time = max(0, interval_seconds - elapsed)
                
                if sleep_time > 0:
                    logger.info(f"Waiting {sleep_time:.1f} seconds until next cycle")
                    
                    # Sleep in small increments to allow for clean shutdown
                    sleep_increment = 1
                    for _ in range(int(sleep_time / sleep_increment)):
                        if self.stop_requested:
                            break
                        time.sleep(sleep_increment)
                    
                    # Sleep any remaining time
                    remaining = sleep_time % sleep_increment
                    if remaining > 0 and not self.stop_requested:
                        time.sleep(remaining)
        
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.is_running = False
            logger.info("Trading bot stopped")
            
    def stop(self):
        """Stop the trading bot gracefully."""
        logger.info("Stopping trading bot...")
        self.stop_requested = True
        
        # Generate final performance report
        self._generate_performance_report(final=True)
        
        # Export trading history
        self.export_trading_history('json')
        
        logger.info("Trading bot stopped")


# =============================================================================
# TRADING VISUALIZER
# =============================================================================

class TradingVisualizer:
    """
    Creates visualizations for trading data and performance analysis.
    Generates charts, dashboards, and exports data for external analysis.
    """
    
    def __init__(self, trading_bot: CryptoTradingBot):
        """Initialize with a trading bot instance."""
        self.trading_bot = trading_bot
        self.output_dir = trading_bot.visualization_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_price_prediction_chart(self):
        """Generate a chart showing price predictions vs actual prices."""
        try:
            # Lazy import matplotlib to avoid overhead when not used
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            # Fetch historical data
            df = self.trading_bot.fetch_historical_data(limit=100)
            
            if df is None or len(df) == 0:
                logger.error("No data available for visualization")
                return None
            
            # Make predictions for multiple points
            predictions = []
            
            # Only try to predict if we have a trained model
            if self.trading_bot.price_predictor.model is not None:
                for i in range(20, len(df)):
                    try:
                        # Use subset of data for prediction
                        temp_df = df.iloc[:i+1]
                        pred = self.trading_bot.price_predictor.predict(temp_df)
                        if pred:
                            predictions.append((df.index[i], pred['predicted_price']))
                    except Exception as e:
                        logger.warning(f"Error predicting at point {i}: {e}")
            
            # Create the chart
            plt.figure(figsize=(12, 8))
            
            # Plot actual prices
            plt.plot(df.index, df['close'], label='Actual Price', color='blue', linewidth=2)
            
            # Plot predictions if available
            if predictions:
                pred_dates, pred_prices = zip(*predictions)
                plt.plot(pred_dates, pred_prices, label='Predicted Price', color='red', linestyle='--', linewidth=2)
            
            # Plot technical indicators
            plt.plot(df.index, df['sma_7'], label='SMA 7', color='green', alpha=0.6)
            plt.plot(df.index, df['sma_25'], label='SMA 25', color='purple', alpha=0.6)
            plt.plot(df.index, df['bb_upper'], label='Bollinger Upper', color='gray', linestyle=':', alpha=0.5)
            plt.plot(df.index, df['bb_lower'], label='Bollinger Lower', color='gray', linestyle=':', alpha=0.5)
            
            # Add trading points
            for trade in self.trading_bot.trades_history:
                try:
                    # Convert ISO string to datetime
                    trade_time = datetime.datetime.fromisoformat(trade['timestamp'])
                    
                    # Only show if within chart range
                    if df.index[0] <= trade_time <= df.index[-1]:
                        marker = '^' if trade['action'] == 'BUY' else 'v'
                        color = 'green' if trade['action'] == 'BUY' else 'red'
                        
                        plt.scatter(trade_time, trade['price'], marker=marker, s=100, 
                                   color=color, zorder=5)
                        
                        plt.annotate(f"{trade['action']}", 
                                    (trade_time, trade['price']), 
                                    textcoords="offset points", 
                                    xytext=(0, 10), 
                                    ha='center', 
                                    fontweight='bold')
                except Exception as e:
                    logger.warning(f"Error plotting trade: {e}")
            
            # Formatting
            plt.title(f'Price Prediction Chart for {self.trading_bot.symbol}', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Price', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            
            # X-axis date formatting
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save the chart
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"price_prediction_{self.trading_bot.symbol.replace('/', '_')}_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=100)
            
            logger.info(f"Price prediction chart saved to {filepath}")
            
            # Close to free memory
            plt.close()
            
            return filepath
            
        except ImportError:
            logger.error("Matplotlib not available for visualization")
            return None
            
        except Exception as e:
            logger.error(f"Error generating price prediction chart: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_performance_dashboard(self):
        """Generate a dashboard of trading performance metrics."""
        try:
            # Lazy import visualization libraries
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            # Check if we have enough data
            if not self.trading_bot.trades_history:
                logger.warning("No trades available for performance dashboard")
                return None
            
            # Extract trade data
            trade_times = []
            cumulative_pnl = []
            actions = []
            prices = []
            running_pnl = 0
            
            # Extract symbol components
            base_currency, quote_currency = self.trading_bot.symbol.split('/')
            
            # Process trades to calculate P&L
            for trade in self.trading_bot.trades_history:
                trade_time = datetime.datetime.fromisoformat(trade['timestamp'])
                trade_times.append(trade_time)
                actions.append(trade['action'])
                prices.append(trade['price'])
                
                # Simple P&L calculation
                if trade['action'] == 'BUY':
                    running_pnl -= trade['value']
                else:  # SELL
                    running_pnl += trade['value']
                
                cumulative_pnl.append(running_pnl)
            
            # Create the dashboard (2x2 grid)
            fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1]})
            
            # 1. Price chart with trades
            ax1 = axes[0]
            
            # Fetch historical price data
            df = self.trading_bot.fetch_historical_data(limit=500)
            
            if df is not None and len(df) > 0:
                # Find overlapping time range
                min_trade_time = min(trade_times)
                max_trade_time = max(trade_times)
                
                # Filter dataframe to relevant time period (with buffer)
                mask = (df.index >= min_trade_time - datetime.timedelta(days=1)) & \
                       (df.index <= max_trade_time + datetime.timedelta(days=1))
                plot_df = df[mask] if mask.any() else df
                
                # Plot price
                ax1.plot(plot_df.index, plot_df['close'], label='Price', color='blue', linewidth=1.5)
                
                # Add trades
                buy_times = [t for t, a in zip(trade_times, actions) if a == 'BUY']
                buy_prices = [p for p, a in zip(prices, actions) if a == 'BUY']
                
                sell_times = [t for t, a in zip(trade_times, actions) if a == 'SELL']
                sell_prices = [p for p, a in zip(prices, actions) if a == 'SELL']
                
                if buy_times:
                    ax1.scatter(buy_times, buy_prices, marker='^', color='green', s=100, label='Buy')
                    
                if sell_times:
                    ax1.scatter(sell_times, sell_prices, marker='v', color='red', s=100, label='Sell')
            else:
                # If no price data, just plot the trade prices
                for i, (time, action, price) in enumerate(zip(trade_times, actions, prices)):
                    color = 'green' if action == 'BUY' else 'red'
                    marker = '^' if action == 'BUY' else 'v'
                    ax1.scatter(time, price, color=color, marker=marker, s=100)
                    
                    # Connect the dots
                    if i > 0:
                        ax1.plot([trade_times[i-1], time], [prices[i-1], price], 
                                color='gray', linestyle='--', alpha=0.5)
            
            # Formatting
            ax1.set_title(f'Trading Activity - {self.trading_bot.symbol}', fontsize=14)
            ax1.set_ylabel(f'Price ({quote_currency})', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            
            # 2. Cumulative P&L chart
            ax2 = axes[1]
            
            # Plot cumulative P&L
            ax2.plot(trade_times, cumulative_pnl, color='purple', linewidth=2, label='Cumulative P&L')
            
            # Add color fill
            ax2.fill_between(trade_times, 0, cumulative_pnl,
                            where=[p > 0 for p in cumulative_pnl],
                            color='green', alpha=0.3, label='Profit')
            ax2.fill_between(trade_times, 0, cumulative_pnl,
                            where=[p <= 0 for p in cumulative_pnl],
                            color='red', alpha=0.3, label='Loss')
            
            # Add zero line
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Formatting
            ax2.set_title('Cumulative Profit/Loss', fontsize=14)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel(f'P&L ({quote_currency})', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper left')
            
            # Format dates on x-axis
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add performance metrics
            performance = self.trading_bot.performance_metrics
            
            # Create text box for metrics
            metrics_text = (
                f"Total Trades: {performance['total_trades']}\n"
                f"Win Rate: {performance['winning_trades'] / performance['total_trades'] * 100:.1f}% "
                f"({performance['winning_trades']}/{performance['total_trades']})\n"
                f"Total P&L: {performance['total_profit']:.2f} {quote_currency} "
                f"({(performance['total_profit'] / performance['start_balance'] * 100) if performance['start_balance'] > 0 else 0:.1f}%)\n"
                f"Max Drawdown: {performance['max_drawdown']:.1f}%"
            ) if performance['total_trades'] > 0 else "No trade data available"
            
            # Position the text box
            props = dict(boxstyle='round', facecolor='white', alpha=0.7)
            fig.text(0.5, 0.01, metrics_text, transform=fig.transFigure, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='center', bbox=props)
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.12)  # Make room for the text box
            
            # Save the dashboard
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_dashboard_{self.trading_bot.symbol.replace('/', '_')}_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=120)
            
            logger.info(f"Performance dashboard saved to {filepath}")
            
            # Close to free memory
            plt.close()
            
            return filepath
            
        except ImportError:
            logger.error("Matplotlib not available for visualization")
            return None
            
        except Exception as e:
            logger.error(f"Error generating performance dashboard: {e}")
            import traceback
            traceback.print_exc()
            return None


# =============================================================================
# GUI APPLICATION (OPTIONAL)
# =============================================================================

def create_gui_application():
    """Create and launch a GUI application for the trading bot."""
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
        
        class TradingBotGUI:
            def __init__(self, root):
                self.root = root
                root.title("Crypto Trading Bot - Optimized for RTX 3080 & i9-13900K")
                root.geometry("1400x800")
                
                # Bot instances
                self.bots = {}
                self.active_threads = {}
                
                # Tab system
                self.tab_control = ttk.Notebook(root)
                
                # Create tabs
                self.setup_tabs()
                
                # Initialize default values
                self.initialize_defaults()
                
                # Log area
                self.setup_log_area()
                
                # Write initial log
                self.log("Crypto Trading Bot started. Optimized for NVIDIA RTX 3080 & Intel i9-13900K")
                self.log("Configure and start the bot using the options above.")
            
            def setup_tabs(self):
                """Set up the tab interface."""
                # Create tabs
                self.tab_config = ttk.Frame(self.tab_control)
                self.tab_monitor = ttk.Frame(self.tab_control)
                self.tab_viz = ttk.Frame(self.tab_control)
                self.tab_system = ttk.Frame(self.tab_control)
                
                # Add tabs to notebook
                self.tab_control.add(self.tab_config, text='Configuration')
                self.tab_control.add(self.tab_monitor, text='Monitor')
                self.tab_control.add(self.tab_viz, text='Visualization')
                self.tab_control.add(self.tab_system, text='System')
                
                # Add notebook to window
                self.tab_control.pack(expand=1, fill="both")
                
                # Set up each tab
                self.setup_config_tab()
                self.setup_monitor_tab()
                self.setup_viz_tab()
                self.setup_system_tab()
            
            def setup_config_tab(self):
                """Set up the configuration tab."""
                # Main configuration frame
                config_frame = ttk.LabelFrame(self.tab_config, text="Trading Bot Configuration")
                config_frame.pack(fill="both", expand=True, padx=10, pady=10)
                
                # Exchange settings
                exchange_frame = ttk.LabelFrame(config_frame, text="Exchange Settings")
                exchange_frame.pack(fill="x", padx=10, pady=10)
                
                ttk.Label(exchange_frame, text="Exchange:").grid(column=0, row=0, padx=10, pady=5, sticky="w")
                self.exchange_var = tk.StringVar()
                ttk.Combobox(exchange_frame, textvariable=self.exchange_var, 
                            values=["binance", "coinbase", "kraken", "kucoin"]).grid(
                    column=1, row=0, padx=10, pady=5)
                
                ttk.Label(exchange_frame, text="API Key:").grid(column=0, row=1, padx=10, pady=5, sticky="w")
                self.api_key_var = tk.StringVar()
                ttk.Entry(exchange_frame, textvariable=self.api_key_var, width=40, show="*").grid(
                    column=1, row=1, padx=10, pady=5)
                
                ttk.Label(exchange_frame, text="API Secret:").grid(column=0, row=2, padx=10, pady=5, sticky="w")
                self.api_secret_var = tk.StringVar()
                ttk.Entry(exchange_frame, textvariable=self.api_secret_var, width=40, show="*").grid(
                    column=1, row=2, padx=10, pady=5)
                
                # Trading settings
                trading_frame = ttk.LabelFrame(config_frame, text="Trading Settings")
                trading_frame.pack(fill="x", padx=10, pady=10)
                
                ttk.Label(trading_frame, text="Symbol:").grid(column=0, row=0, padx=10, pady=5, sticky="w")
                self.symbol_var = tk.StringVar()
                ttk.Combobox(trading_frame, textvariable=self.symbol_var, 
                            values=["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]).grid(
                    column=1, row=0, padx=10, pady=5)
                
                ttk.Label(trading_frame, text="Timeframe:").grid(column=0, row=1, padx=10, pady=5, sticky="w")
                self.timeframe_var = tk.StringVar()
                ttk.Combobox(trading_frame, textvariable=self.timeframe_var, 
                            values=["1m", "5m", "15m", "30m", "1h", "4h", "1d"]).grid(
                    column=1, row=1, padx=10, pady=5)
                
                ttk.Label(trading_frame, text="Trade Size (%):").grid(column=0, row=2, padx=10, pady=5, sticky="w")
                self.trade_amount_var = tk.StringVar()
                ttk.Entry(trading_frame, textvariable=self.trade_amount_var).grid(
                    column=1, row=2, padx=10, pady=5)
                
                ttk.Label(trading_frame, text="Stop Loss (%):").grid(column=0, row=3, padx=10, pady=5, sticky="w")
                self.stop_loss_var = tk.StringVar()
                ttk.Entry(trading_frame, textvariable=self.stop_loss_var).grid(
                    column=1, row=3, padx=10, pady=5)
                
                ttk.Label(trading_frame, text="Take Profit (%):").grid(column=0, row=4, padx=10, pady=5, sticky="w")
                self.take_profit_var = tk.StringVar()
                ttk.Entry(trading_frame, textvariable=self.take_profit_var).grid(
                    column=1, row=4, padx=10, pady=5)
                
                # ML settings
                ml_frame = ttk.LabelFrame(config_frame, text="Machine Learning Settings")
                ml_frame.pack(fill="x", padx=10, pady=10)
                
                ttk.Label(ml_frame, text="Training Interval (hours):").grid(column=0, row=0, padx=10, pady=5, sticky="w")
                self.train_interval_var = tk.StringVar()
                ttk.Entry(ml_frame, textvariable=self.train_interval_var).grid(
                    column=1, row=0, padx=10, pady=5)
                
                ttk.Label(ml_frame, text="Use GPU Acceleration:").grid(column=0, row=1, padx=10, pady=5, sticky="w")
                self.use_gpu_var = tk.BooleanVar(value=True)
                ttk.Checkbutton(ml_frame, variable=self.use_gpu_var).grid(
                    column=1, row=1, padx=10, pady=5, sticky="w")
                
                # Control buttons
                button_frame = ttk.Frame(config_frame)
                button_frame.pack(fill="x", padx=10, pady=20)
                
                ttk.Button(button_frame, text="Start Bot", command=self.start_bot).grid(
                    column=0, row=0, padx=10, pady=5)
                
                ttk.Button(button_frame, text="Stop Bot", command=self.stop_bot).grid(
                    column=1, row=0, padx=10, pady=5)
                
                ttk.Button(button_frame, text="Test Connection", command=self.test_connection).grid(
                    column=2, row=0, padx=10, pady=5)
            
            def setup_monitor_tab(self):
                """Set up the monitoring tab."""
                # Main monitoring frame
                monitor_frame = ttk.LabelFrame(self.tab_monitor, text="Live Trading Monitor")
                monitor_frame.pack(fill="both", expand=True, padx=10, pady=10)
                
                # Price and prediction info
                info_frame = ttk.LabelFrame(monitor_frame, text="Market Information")
                info_frame.pack(fill="x", padx=10, pady=10)
                
                # Current price
                ttk.Label(info_frame, text="Current Price:").grid(column=0, row=0, padx=10, pady=5, sticky="w")
                self.current_price_var = tk.StringVar(value="--")
                ttk.Label(info_frame, textvariable=self.current_price_var, font=("Arial", 12, "bold")).grid(
                    column=1, row=0, padx=10, pady=5, sticky="w")
                
                # Prediction
                ttk.Label(info_frame, text="Predicted Price:").grid(column=0, row=1, padx=10, pady=5, sticky="w")
                self.predicted_price_var = tk.StringVar(value="--")
                ttk.Label(info_frame, textvariable=self.predicted_price_var).grid(
                    column=1, row=1, padx=10, pady=5, sticky="w")
                                # Sentiment
                ttk.Label(info_frame, text="Market Sentiment:").grid(column=0, row=2, padx=10, pady=5, sticky="w")
                self.sentiment_var = tk.StringVar(value="--")
                ttk.Label(info_frame, textvariable=self.sentiment_var).grid(
                    column=1, row=2, padx=10, pady=5, sticky="w")
                
                # Balance info
                ttk.Label(info_frame, text="Balance:").grid(column=2, row=0, padx=10, pady=5, sticky="w")
                self.balance_var = tk.StringVar(value="--")
                ttk.Label(info_frame, textvariable=self.balance_var, font=("Arial", 12, "bold")).grid(
                    column=3, row=0, padx=10, pady=5, sticky="w")
                
                # P&L
                ttk.Label(info_frame, text="Total P&L:").grid(column=2, row=1, padx=10, pady=5, sticky="w")
                self.pnl_var = tk.StringVar(value="--")
                ttk.Label(info_frame, textvariable=self.pnl_var).grid(
                    column=3, row=1, padx=10, pady=5, sticky="w")
                
                # Win Rate
                ttk.Label(info_frame, text="Win Rate:").grid(column=2, row=2, padx=10, pady=5, sticky="w")
                self.win_rate_var = tk.StringVar(value="--")
                ttk.Label(info_frame, textvariable=self.win_rate_var).grid(
                    column=3, row=2, padx=10, pady=5, sticky="w")
                
                # Trades table
                trades_frame = ttk.LabelFrame(monitor_frame, text="Recent Trades")
                trades_frame.pack(fill="both", expand=True, padx=10, pady=10)
                
                # Table columns
                columns = ('time', 'action', 'amount', 'price', 'value')
                self.trades_table = ttk.Treeview(trades_frame, columns=columns, show='headings')
                
                # Define headings
                self.trades_table.heading('time', text='Time')
                self.trades_table.heading('action', text='Action')
                self.trades_table.heading('amount', text='Amount')
                self.trades_table.heading('price', text='Price')
                self.trades_table.heading('value', text='Value')
                
                # Set column widths
                self.trades_table.column('time', width=150)
                self.trades_table.column('action', width=70)
                self.trades_table.column('amount', width=100)
                self.trades_table.column('price', width=100)
                self.trades_table.column('value', width=100)
                
                # Add scrollbar
                scrollbar = ttk.Scrollbar(trades_frame, orient=tk.VERTICAL, command=self.trades_table.yview)
                self.trades_table.configure(yscroll=scrollbar.set)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                self.trades_table.pack(fill="both", expand=True, padx=5, pady=5)
                
                # Refresh button
                ttk.Button(monitor_frame, text="Refresh Data", command=self.update_monitor).pack(
                    pady=10)
            
            def setup_viz_tab(self):
                """Set up the visualization tab."""
                # Main visualization frame
                viz_frame = ttk.LabelFrame(self.tab_viz, text="Trading Visualizations")
                viz_frame.pack(fill="both", expand=True, padx=10, pady=10)
                
                # Chart controls
                control_frame = ttk.Frame(viz_frame)
                control_frame.pack(fill="x", padx=10, pady=10)
                
                ttk.Button(control_frame, text="Price Prediction Chart", 
                          command=self.generate_price_chart).pack(side="left", padx=5)
                
                ttk.Button(control_frame, text="Performance Dashboard", 
                          command=self.generate_dashboard).pack(side="left", padx=5)
                
                ttk.Button(control_frame, text="Export Trading History", 
                          command=self.export_history).pack(side="left", padx=5)
                
                # Image display area
                self.image_frame = ttk.LabelFrame(viz_frame, text="Visualization Output")
                self.image_frame.pack(fill="both", expand=True, padx=10, pady=10)
                
                ttk.Label(self.image_frame, text="Generate a chart to display here").pack(
                    expand=True, pady=100)
            
            def setup_system_tab(self):
                """Set up the system information tab."""
                # Main system frame
                system_frame = ttk.LabelFrame(self.tab_system, text="System Information")
                system_frame.pack(fill="both", expand=True, padx=10, pady=10)
                
                # Hardware info
                hw_frame = ttk.LabelFrame(system_frame, text="Hardware Information")
                hw_frame.pack(fill="x", padx=10, pady=10)
                
                # GPU accelerator for info
                self.gpu_accelerator = GPUAccelerator()
                gpu_info = self.gpu_accelerator.get_status()
                
                # CPU info
                ttk.Label(hw_frame, text="CPU:").grid(column=0, row=0, padx=10, pady=5, sticky="w")
                ttk.Label(hw_frame, text="Intel i9-13900K").grid(column=1, row=0, padx=10, pady=5, sticky="w")
                
                # GPU info
                ttk.Label(hw_frame, text="GPU:").grid(column=0, row=1, padx=10, pady=5, sticky="w")
                gpu_text = "NVIDIA RTX 3080" if gpu_info.get('available', False) else "No GPU detected"
                ttk.Label(hw_frame, text=gpu_text).grid(column=1, row=1, padx=10, pady=5, sticky="w")
                
                # RAM info
                ttk.Label(hw_frame, text="RAM:").grid(column=0, row=2, padx=10, pady=5, sticky="w")
                ttk.Label(hw_frame, text="32 GB").grid(column=1, row=2, padx=10, pady=5, sticky="w")
                
                # Python & library versions
                versions_frame = ttk.LabelFrame(system_frame, text="Software Versions")
                versions_frame.pack(fill="x", padx=10, pady=10)
                
                ttk.Label(versions_frame, text="Python:").grid(column=0, row=0, padx=10, pady=5, sticky="w")
                ttk.Label(versions_frame, text=f"{sys.version.split(' ')[0]}").grid(
                    column=1, row=0, padx=10, pady=5, sticky="w")
                
                ttk.Label(versions_frame, text="TensorFlow:").grid(column=0, row=1, padx=10, pady=5, sticky="w")
                ttk.Label(versions_frame, text=f"{tf.__version__}").grid(
                    column=1, row=1, padx=10, pady=5, sticky="w")
                
                # System monitoring
                monitor_frame = ttk.LabelFrame(system_frame, text="Resource Monitoring")
                monitor_frame.pack(fill="x", padx=10, pady=10)
                
                ttk.Label(monitor_frame, text="CPU Usage:").grid(column=0, row=0, padx=10, pady=5, sticky="w")
                self.cpu_usage_var = tk.StringVar(value="--")
                ttk.Label(monitor_frame, textvariable=self.cpu_usage_var).grid(
                    column=1, row=0, padx=10, pady=5, sticky="w")
                
                ttk.Label(monitor_frame, text="GPU Usage:").grid(column=0, row=1, padx=10, pady=5, sticky="w")
                self.gpu_usage_var = tk.StringVar(value="--")
                ttk.Label(monitor_frame, textvariable=self.gpu_usage_var).grid(
                    column=1, row=1, padx=10, pady=5, sticky="w")
                
                ttk.Label(monitor_frame, text="Memory Usage:").grid(column=0, row=2, padx=10, pady=5, sticky="w")
                self.memory_usage_var = tk.StringVar(value="--")
                ttk.Label(monitor_frame, textvariable=self.memory_usage_var).grid(
                    column=1, row=2, padx=10, pady=5, sticky="w")
                
                # Refresh button
                ttk.Button(monitor_frame, text="Refresh", command=self.update_system_info).grid(
                    column=0, row=3, columnspan=2, padx=10, pady=10)
            
            def setup_log_area(self):
                """Set up the logging area."""
                log_frame = ttk.LabelFrame(self.root, text="Log Output")
                log_frame.pack(fill="both", expand=True, padx=10, pady=5)
                
                self.log_text = tk.Text(log_frame, height=10)
                self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
                
                # Add scrollbar
                scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
                scrollbar.pack(side="right", fill="y")
                self.log_text.config(yscrollcommand=scrollbar.set)
            
            def initialize_defaults(self):
                """Initialize default values for UI controls."""
                self.exchange_var.set("binance")
                self.api_key_var.set("")
                self.api_secret_var.set("")
                self.symbol_var.set("BTC/USDT")
                self.timeframe_var.set("1h")
                self.trade_amount_var.set("2.0")
                self.stop_loss_var.set("2.0")
                self.take_profit_var.set("5.0")
                self.train_interval_var.set("12")
            
            def log(self, message):
                """Add a message to the log area."""
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.log_text.insert("end", f"[{timestamp}] {message}\n")
                self.log_text.see("end")
            
            def start_bot(self):
                """Start the trading bot."""
                try:
                    # Get configuration values
                    exchange = self.exchange_var.get()
                    api_key = self.api_key_var.get()
                    api_secret = self.api_secret_var.get()
                    symbol = self.symbol_var.get()
                    timeframe = self.timeframe_var.get()
                    
                    # Convert percentages to decimal
                    trade_amount = float(self.trade_amount_var.get()) / 100.0
                    stop_loss = float(self.stop_loss_var.get()) / 100.0
                    take_profit = float(self.take_profit_var.get()) / 100.0
                    
                    training_interval = int(self.train_interval_var.get())
                    
                    # Create bot ID from symbol and timeframe
                    bot_id = f"{symbol.replace('/', '_')}_{timeframe}"
                    
                    # Check if bot already running
                    if bot_id in self.active_threads and self.active_threads[bot_id].is_alive():
                        messagebox.showwarning("Bot Active", f"Trading bot for {symbol} is already running!")
                        return
                    
                    # Create bot instance
                    self.log(f"Creating trading bot for {symbol} on {exchange}...")
                    
                    bot = CryptoTradingBot(
                        api_key=api_key,
                        api_secret=api_secret,
                        symbol=symbol,
                        timeframe=timeframe,
                        exchange=exchange
                    )
                    
                    # Configure bot settings
                    bot.trade_amount = trade_amount
                    bot.stop_loss = stop_loss
                    bot.take_profit = take_profit
                    
                    # Set GPU settings
                    if not self.use_gpu_var.get():
                        self.log("Disabling GPU acceleration as per configuration")
                        import os
                        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                    
                    # Store bot instance
                    self.bots[bot_id] = bot
                    
                    # Start bot in a separate thread
                    bot_thread = threading.Thread(
                        target=bot.run,
                        args=(60, training_interval),
                        name=f"Bot-{bot_id}"
                    )
                    bot_thread.daemon = True
                    
                    # Start the thread
                    bot_thread.start()
                    self.active_threads[bot_id] = bot_thread
                    
                    self.log(f"Trading bot started for {symbol}")
                    messagebox.showinfo("Bot Started", f"Trading bot for {symbol} has been started successfully!")
                    
                    # Switch to monitoring tab
                    self.tab_control.select(self.tab_monitor)
                    
                    # Start periodic updates
                    self.schedule_update()
                    
                except Exception as e:
                    self.log(f"Error starting bot: {e}")
                    messagebox.showerror("Error", f"Failed to start bot: {e}")
            
            def stop_bot(self):
                """Stop the trading bot."""
                try:
                    # Find active bot
                    symbol = self.symbol_var.get()
                    timeframe = self.timeframe_var.get()
                    bot_id = f"{symbol.replace('/', '_')}_{timeframe}"
                    
                    if bot_id in self.bots:
                        # Get bot instance
                        bot = self.bots[bot_id]
                        
                        # Request stop
                        self.log(f"Stopping trading bot for {symbol}...")
                        bot.stop()
                        
                        # Wait for thread to complete
                        if bot_id in self.active_threads:
                            self.active_threads[bot_id].join(timeout=5)
                            del self.active_threads[bot_id]
                        
                        # Remove bot from list
                        del self.bots[bot_id]
                        
                        self.log(f"Trading bot for {symbol} stopped")
                        messagebox.showinfo("Bot Stopped", f"Trading bot for {symbol} has been stopped")
                    else:
                        messagebox.showwarning("No Bot", f"No active trading bot found for {symbol}")
                
                except Exception as e:
                    self.log(f"Error stopping bot: {e}")
                    messagebox.showerror("Error", f"Failed to stop bot: {e}")
            
            def test_connection(self):
                """Test the connection to the exchange."""
                try:
                    exchange = self.exchange_var.get()
                    api_key = self.api_key_var.get()
                    api_secret = self.api_secret_var.get()
                    
                    self.log(f"Testing connection to {exchange}...")
                    
                    # Create exchange manager
                    manager = ExchangeAPIManager(exchange, api_key, api_secret)
                    if manager.initialize():
                        self.log(f"Successfully connected to {exchange}")
                        
                        # Check balance if API keys provided
                        if api_key and api_secret:
                            balance = manager.fetch_balance()
                            
                            if balance:
                                self.log("Connection with API credentials verified successfully!")
                                
                                # Show some balance details
                                currencies = ["USDT", "BTC", "ETH"]
                                balances = []
                                
                                for currency in currencies:
                                    amount = balance['total'].get(currency, 0)
                                    if amount > 0:
                                        balances.append(f"{amount} {currency}")
                                
                                if balances:
                                    self.log(f"Balances: {', '.join(balances)}")
                                
                                messagebox.showinfo("Connection Successful", 
                                                  f"Successfully connected to {exchange} with API credentials!")
                            else:
                                self.log("Connection succeeded, but could not fetch balance")
                                messagebox.showwarning("Partial Success", 
                                                    f"Connected to {exchange}, but could not fetch balance. Check API permissions.")
                        else:
                            self.log("Connection succeeded (public API only)")
                            messagebox.showinfo("Connection Successful", 
                                              f"Successfully connected to {exchange} (public API only)")
                    else:
                        self.log(f"Failed to connect to {exchange}")
                        messagebox.showerror("Connection Failed", f"Failed to connect to {exchange}")
                
                except Exception as e:
                    self.log(f"Connection test error: {e}")
                    messagebox.showerror("Error", f"Connection test failed: {e}")
            
            def update_monitor(self):
                """Update the monitoring display."""
                # Check if any bot is active
                if not self.bots:
                    return
                
                # Get current bot
                symbol = self.symbol_var.get()
                timeframe = self.timeframe_var.get()
                bot_id = f"{symbol.replace('/', '_')}_{timeframe}"
                
                if bot_id not in self.bots:
                    return
                
                bot = self.bots[bot_id]
                
                try:
                    # Update price display
                    current_price = bot.exchange_manager.get_current_price(symbol)
                    if current_price:
                        self.current_price_var.set(f"{current_price:.8f}")
                        
                    # Update prediction
                    prediction = bot.predict_next_price()
                    if prediction:
                        self.predicted_price_var.set(f"{prediction['predicted_price']:.8f} ({prediction['price_change_percent']:.2f}%)")
                        
                    # Update sentiment
                    sentiment = bot.get_market_sentiment()
                    if sentiment:
                        self.sentiment_var.set(f"{sentiment['overall'].upper()} ({sentiment['strength']:.2f})")
                        
                    # Update balance
                    balance_info = bot.check_balance()
                    if balance_info:
                        _, quote_currency = symbol.split('/')
                        quote_balance = balance_info.get(quote_currency, 0)
                        self.balance_var.set(f"{quote_balance:.2f} {quote_currency}")
                        
                        # Update P&L
                        if bot.performance_metrics['start_balance'] > 0:
                            pnl = bot.performance_metrics['total_profit']
                            pnl_pct = (pnl / bot.performance_metrics['start_balance']) * 100
                            self.pnl_var.set(f"{pnl:.2f} {quote_currency} ({pnl_pct:.2f}%)")
                            
                        # Update win rate
                        if bot.performance_metrics['total_trades'] > 0:
                            win_rate = (bot.performance_metrics['winning_trades'] / bot.performance_metrics['total_trades']) * 100
                            self.win_rate_var.set(f"{win_rate:.1f}% ({bot.performance_metrics['winning_trades']}/{bot.performance_metrics['total_trades']})")
                            
                    # Update trades table
                    self.trades_table.delete(*self.trades_table.get_children())
                    
                    # Add recent trades (last 10)
                    for trade in bot.trades_history[-10:]:
                        time_str = datetime.datetime.fromisoformat(trade['timestamp']).strftime("%Y-%m-%d %H:%M")
                        
                        self.trades_table.insert('', 'end', values=(
                            time_str,
                            trade['action'],
                            f"{trade['amount']:.8f}",
                            f"{trade['price']:.8f}",
                            f"{trade['value']:.2f}"
                        ))
                    
                    self.log(f"Monitor updated for {symbol}")
                    
                except Exception as e:
                    self.log(f"Error updating monitor: {e}")
            
            def schedule_update(self):
                """Schedule a periodic update of the monitoring display."""
                self.update_monitor()
                # Schedule next update in 10 seconds
                self.root.after(10000, self.schedule_update)
            
            def update_system_info(self):
                """Update system resource usage information."""
                try:
                    import psutil
                    
                    # Update CPU usage
                    cpu_percent = psutil.cpu_percent(interval=0.5)
                    self.cpu_usage_var.set(f"{cpu_percent}%")
                    
                    # Update memory usage
                    memory = psutil.virtual_memory()
                    self.memory_usage_var.set(f"{memory.percent}% ({memory.used / (1024**3):.1f} GB)")
                    
                    # Update GPU usage if available
                    gpu_info = self.gpu_accelerator.get_status()
                    
                    if gpu_info.get('available', False):
                        if 'gpu_utilization' in gpu_info:
                            self.gpu_usage_var.set(f"{gpu_info['gpu_utilization']}%")
                        else:
                            self.gpu_usage_var.set("Available (usage unknown)")
                    else:
                        self.gpu_usage_var.set("Not available")
                    
                    self.log("System information updated")
                    
                except ImportError:
                    self.log("psutil module not available for system monitoring")
                    
                except Exception as e:
                    self.log(f"Error updating system info: {e}")
            
            def generate_price_chart(self):
                """Generate and display a price prediction chart."""
                # Check if any bot is active
                if not self.bots:
                    messagebox.showwarning("No Bot", "No active trading bot found")
                    return
                
                # Get current bot
                symbol = self.symbol_var.get()
                timeframe = self.timeframe_var.get()
                bot_id = f"{symbol.replace('/', '_')}_{timeframe}"
                
                if bot_id not in self.bots:
                    messagebox.showwarning("No Bot", f"No active trading bot found for {symbol}")
                    return
                
                bot = self.bots[bot_id]
                
                # Create visualizer
                visualizer = TradingVisualizer(bot)
                
                # Generate chart
                self.log("Generating price prediction chart...")
                chart_path = visualizer.generate_price_prediction_chart()
                
                if chart_path:
                    # Clear previous content
                    for widget in self.image_frame.winfo_children():
                        widget.destroy()
                    
                    # Load and display image
                    try:
                        from PIL import Image, ImageTk
                        
                        img = Image.open(chart_path)
                        img = img.resize((800, 500), Image.LANCZOS)
                        photo = ImageTk.PhotoImage(img)
                        
                        label = ttk.Label(self.image_frame, image=photo)
                        label.image = photo  # Keep a reference
                        label.pack(fill="both", expand=True)
                        
                        self.log(f"Chart displayed from {chart_path}")
                    except ImportError:
                        self.log("PIL module not available for image display")
                        ttk.Label(self.image_frame, text=f"Chart saved to {chart_path}").pack(expand=True)
                else:
                    self.log("Failed to generate chart")
            
            def generate_dashboard(self):
                """Generate and display a performance dashboard."""
                # Check if any bot is active
                if not self.bots:
                    messagebox.showwarning("No Bot", "No active trading bot found")
                    return
                
                # Get current bot
                symbol = self.symbol_var.get()
                timeframe = self.timeframe_var.get()
                bot_id = f"{symbol.replace('/', '_')}_{timeframe}"
                
                if bot_id not in self.bots:
                    messagebox.showwarning("No Bot", f"No active trading bot found for {symbol}")
                    return
                
                bot = self.bots[bot_id]
                
                # Create visualizer
                visualizer = TradingVisualizer(bot)
                
                # Generate dashboard
                self.log("Generating performance dashboard...")
                dashboard_path = visualizer.generate_performance_dashboard()
                
                if dashboard_path:
                    # Clear previous content
                    for widget in self.image_frame.winfo_children():
                        widget.destroy()
                    
                    # Load and display image
                    try:
                        from PIL import Image, ImageTk
                        
                        img = Image.open(dashboard_path)
                        img = img.resize((800, 600), Image.LANCZOS)
                        photo = ImageTk.PhotoImage(img)
                        
                        label = ttk.Label(self.image_frame, image=photo)
                        label.image = photo  # Keep a reference
                        label.pack(fill="both", expand=True)
                        
                        self.log(f"Dashboard displayed from {dashboard_path}")
                    except ImportError:
                        self.log("PIL module not available for image display")
                        ttk.Label(self.image_frame, text=f"Dashboard saved to {dashboard_path}").pack(expand=True)
                else:
                    self.log("Failed to generate dashboard")
            
            def export_history(self):
                """Export trading history to a file."""
                # Check if any bot is active
                if not self.bots:
                    messagebox.showwarning("No Bot", "No active trading bot found")
                    return
                
                # Get current bot
                symbol = self.symbol_var.get()
                timeframe = self.timeframe_var.get()
                bot_id = f"{symbol.replace('/', '_')}_{timeframe}"
                
                if bot_id not in self.bots:
                    messagebox.showwarning("No Bot", f"No active trading bot found for {symbol}")
                    return
                
                bot = self.bots[bot_id]
                
                # Ask for format
                formats = ["JSON", "CSV"]
                format_var = tk.StringVar(value="JSON")
                
                # Create a simple dialog
                dialog = tk.Toplevel(self.root)
                dialog.title("Export Format")
                dialog.transient(self.root)
                dialog.grab_set()
                
                ttk.Label(dialog, text="Select export format:").pack(padx=20, pady=10)
                
                for fmt in formats:
                    ttk.Radiobutton(dialog, text=fmt, variable=format_var, value=fmt).pack(anchor="w", padx=20)
                
                def on_ok():
                    dialog.result = format_var.get()
                    dialog.destroy()
                
                ttk.Button(dialog, text="Export", command=on_ok).pack(pady=20)
                
                # Center dialog
                dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50,
                                           self.root.winfo_rooty() + 50))
                
                # Wait for dialog
                self.root.wait_window(dialog)
                
                if hasattr(dialog, 'result'):
                    export_format = dialog.result.lower()
                    
                    # Export history
                    self.log(f"Exporting trading history as {export_format}...")
                    filepath = bot.export_trading_history(export_format)
                    
                    if filepath:
                        self.log(f"Trading history exported to {filepath}")
                        messagebox.showinfo("Export Successful", f"Trading history exported to {filepath}")
                    else:
                        self.log("Failed to export trading history")
                        messagebox.showerror("Export Failed", "Failed to export trading history")
        
        # Create and run the GUI
        root = tk.Tk()
        app = TradingBotGUI(root)
        root.mainloop()
        
    except ImportError:
        logger.error("tkinter not available for GUI application")
        print("GUI application requires tkinter, which is not available.")
        print("You can still use the bot programmatically.")
        return None


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def run_cli():
    """Run the trading bot from command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Crypto Trading Bot with AI/ML')
    
    # Required arguments
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g. BTC/USDT)')
    parser.add_argument('--exchange', type=str, required=True, help='Exchange name (e.g. binance)')
    
    # Optional arguments
    parser.add_argument('--timeframe', type=str, default='1h', help='Trading timeframe (default: 1h)')
    parser.add_argument('--api-key', type=str, help='API key for the exchange')
    parser.add_argument('--api-secret', type=str, help='API secret for the exchange')
    parser.add_argument('--trade-amount', type=float, default=2.0, help='Trade amount in percent (default: 2.0%%)')
    parser.add_argument('--stop-loss', type=float, default=2.0, help='Stop loss in percent (default: 2.0%%)')
    parser.add_argument('--take-profit', type=float, default=5.0, help='Take profit in percent (default: 5.0%%)')
    parser.add_argument('--training-interval', type=int, default=12, help='Model training interval in hours (default: 12)')
    parser.add_argument('--interval', type=int, default=60, help='Trading interval in seconds (default: 60)')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode without real trading')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    
    args = parser.parse_args()
    
    # Process arguments
    symbol = args.symbol
    exchange = args.exchange
    timeframe = args.timeframe
    api_key = args.api_key if not args.demo else None
    api_secret = args.api_secret if not args.demo else None
    trade_amount = args.trade_amount / 100.0
    stop_loss = args.stop_loss / 100.0
    take_profit = args.take_profit / 100.0
    training_interval = args.training_interval
    interval = args.interval
    
        # Disable GPU if requested
    if args.no_gpu:
        print("GPU acceleration disabled")
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Create and configure the bot
    bot = CryptoTradingBot(
        api_key=api_key,
        api_secret=api_secret,
        symbol=symbol,
        timeframe=timeframe,
        exchange=exchange
    )
    
    # Set trading parameters
    bot.trade_amount = trade_amount
    bot.stop_loss = stop_loss
    bot.take_profit = take_profit
    
    # Print configuration
    print(f"Trading Bot Configuration:")
    print(f"Symbol: {symbol}")
    print(f"Exchange: {exchange}")
    print(f"Timeframe: {timeframe}")
    print(f"Trade Amount: {trade_amount*100:.1f}%")
    print(f"Stop Loss: {stop_loss*100:.1f}%")
    print(f"Take Profit: {take_profit*100:.1f}%")
    print(f"Training Interval: {training_interval} hours")
    print(f"Trading Interval: {interval} seconds")
    print(f"Demo Mode: {'Enabled' if args.demo else 'Disabled'}")
    print(f"GPU Acceleration: {'Disabled' if args.no_gpu else 'Enabled'}")
    
    # Run the bot
    try:
        bot.run(interval_seconds=interval, training_interval_hours=training_interval)
    except KeyboardInterrupt:
        print("\nTrading bot stopped by user")
        bot.stop()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Print banner
    print("=" * 80)
    print(" Crypto Trading Bot - Optimized for NVIDIA RTX 3080 & Intel i9-13900K")
    print("=" * 80)
    print(" Version 1.2.0")
    print(" Hardware: NVIDIA RTX 3080, Intel i9-13900K, 32GB RAM")
    print(" Author: BeoWulf5011")
    print(" Last updated: 2025-02-26")
    print("=" * 80)
    print("")
    
    # Check for GUI mode flag
    if "--gui" in sys.argv:
        print("Starting in GUI mode...")
        create_gui_application()
    else:
        # Run in CLI mode
        run_cli()
