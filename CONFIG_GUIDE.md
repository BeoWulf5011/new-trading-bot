# Trading Bot Configuration Guide

## Quick Parameter Reference

### 🎯 Key Trading Parameters

```python
TRADING_PARAMS = {
    'default_trade_amount': 0.02,          # 2% of balance per trade
    'default_stop_loss': 0.015,            # 1.5% stop loss (IMPROVED from 2%)
    'default_take_profit': 0.06,           # 6% take profit (IMPROVED from 5%)
    'trailing_stop_loss': 0.01,            # 1% trailing stop (NEW)
    'min_quote_reserve': 5.0,              # Keep $5 as reserve
    'max_position_size': 0.10,             # Max 10% per trade (NEW)
    'min_confidence_threshold': 0.60,      # Min 60% ML confidence (NEW)
    'trend_strength_threshold': 0.65,      # Min 65% trend strength (NEW)
}
```

### 🤖 ML Model Configuration

```python
ML_CONFIG = {
    'lstm_units': [128, 256, 128],         # LSTM layer sizes
    'dropout_rate': 0.3,                   # IMPROVED from 0.2
    'learning_rate': 0.0005,               # IMPROVED from 0.001
    'early_stopping_patience': 15,         # IMPROVED from 10
    'reduce_lr_patience': 5,
    'batch_size': 64,
    'epochs': 150,                         # IMPROVED from 100
    'lookback_period': 60,                 # 60 time steps history
    'prediction_horizon': 24,              # Predict 24 hours ahead
}
```

## 🔧 Tuning Guide

### For Conservative Trading (Lower Risk)
```python
TRADING_PARAMS = {
    'default_trade_amount': 0.01,          # 1% per trade
    'default_stop_loss': 0.01,             # 1% stop loss
    'default_take_profit': 0.04,           # 4% take profit
    'trailing_stop_loss': 0.008,           # 0.8% trailing
    'max_position_size': 0.05,             # Max 5% per trade
    'min_confidence_threshold': 0.70,      # Require 70% confidence
    'trend_strength_threshold': 0.75,      # Require 75% trend strength
}
```

### For Aggressive Trading (Higher Risk/Reward)
```python
TRADING_PARAMS = {
    'default_trade_amount': 0.05,          # 5% per trade
    'default_stop_loss': 0.02,             # 2% stop loss
    'default_take_profit': 0.10,           # 10% take profit
    'trailing_stop_loss': 0.015,           # 1.5% trailing
    'max_position_size': 0.20,             # Max 20% per trade
    'min_confidence_threshold': 0.55,      # Accept 55% confidence
    'trend_strength_threshold': 0.60,      # Accept 60% trend strength
}
```

### For High Volatility Markets (Crypto Bull Market)
```python
TRADING_PARAMS = {
    'default_trade_amount': 0.015,         # 1.5% per trade
    'default_stop_loss': 0.025,            # 2.5% stop loss (wider)
    'default_take_profit': 0.08,           # 8% take profit (higher)
    'trailing_stop_loss': 0.02,            # 2% trailing (wider)
    'max_position_size': 0.08,             # Max 8% per trade
    'high_volatility_threshold': 0.7,      # Lower threshold
    'min_confidence_threshold': 0.65,      # Higher confidence needed
}
```

### For Stable/Low Volatility Markets
```python
TRADING_PARAMS = {
    'default_trade_amount': 0.03,          # 3% per trade
    'default_stop_loss': 0.01,             # 1% stop loss (tighter)
    'default_take_profit': 0.04,           # 4% take profit (lower)
    'trailing_stop_loss': 0.008,           # 0.8% trailing (tighter)
    'max_position_size': 0.15,             # Max 15% per trade
    'high_volatility_threshold': 0.9,      # Higher threshold
    'min_confidence_threshold': 0.58,      # Slightly lower confidence OK
}
```

## 📊 Performance Metrics Interpretation

### Sharpe Ratio
- **< 0**: Negative risk-adjusted returns (bad)
- **0-1**: Below average returns for risk taken
- **1-2**: Good risk-adjusted returns
- **> 2**: Excellent risk-adjusted returns
- **> 3**: Exceptional (very rare)

### Sortino Ratio (similar to Sharpe but only downside)
- **> Sharpe Ratio**: Good (means upside volatility dominates)
- **< Sharpe Ratio**: Concerning (more downside volatility)

### Calmar Ratio (Return / Max Drawdown)
- **> 0.5**: Acceptable
- **> 1.0**: Good
- **> 2.0**: Excellent
- **> 3.0**: Outstanding

### Profit Factor (Total Gains / Total Losses)
- **< 1.0**: Losing strategy
- **1.0-1.5**: Marginal
- **1.5-2.0**: Good
- **> 2.0**: Excellent
- **> 3.0**: Outstanding

### Win Rate
- **< 50%**: Need high win/loss ratio to be profitable
- **50-60%**: Good if combined with proper R:R
- **60-70%**: Very good
- **> 70%**: Excellent (but verify it's not overfitting)

### Win/Loss Ratio (Avg Win / Avg Loss)
- **< 1.0**: Wins smaller than losses (need high win rate)
- **1.0-1.5**: Balanced
- **1.5-2.5**: Good
- **> 2.5**: Excellent

## 🎨 Volatility Thresholds

```python
'high_volatility_threshold': 0.8    # Above this = high volatility
'med_volatility_threshold': 0.5     # Above this = medium volatility
```

**Effects on Position Sizing:**
- **High volatility (>0.8)**: Position size reduced to 40% of base
- **Medium volatility (0.5-0.8)**: Position size reduced to 70% of base
- **Low volatility (<0.5)**: Full position size (100% of base)

## 🚦 Signal Confidence Requirements

### Strong Buy Signal Requirements:
- Bullish sentiment (>0.65 strength)
- ML prediction bullish (>0.75 confidence)
- Predicted price increase >2%
- All indicators must align

### Strong Sell Signal Requirements:
- Bearish sentiment (>0.65 strength)
- ML prediction bearish (>0.75 confidence)
- Predicted price decrease <-2%
- All indicators must align

### Regular Buy/Sell:
- Sentiment strength >0.55
- Predicted price change >0.5%
- ML confidence >0.60
- No major conflicting signals

## 💡 Optimization Tips

### 1. Finding Your Risk Tolerance
Start conservative and gradually increase:
```python
Week 1: 1% trades, 1% stop loss
Week 2: 1.5% trades, 1.2% stop loss
Week 3: 2% trades, 1.5% stop loss
```

### 2. Confidence Threshold Tuning
- Too high (>0.75): Misses opportunities, fewer trades
- Too low (<0.50): Too many trades, lower quality
- Optimal: 0.60-0.70 for most markets

### 3. Take Profit vs Stop Loss Ratio
- Conservative: 2:1 (6% TP, 3% SL)
- Balanced: 3:1 (6% TP, 2% SL)
- Aggressive: 4:1 (6% TP, 1.5% SL)
- Current: 4:1 (6% TP, 1.5% SL)

### 4. Training Frequency
- High volatility: Train every 6-8 hours
- Normal conditions: Train every 12 hours (default)
- Stable markets: Train every 24 hours

### 5. Data Requirements
- **Minimum**: 200 data points for basic training
- **Recommended**: 1000-2000 data points
- **Optimal**: 2000+ data points for best patterns

## 🎯 Asset-Specific Recommendations

### Bitcoin (BTC)
```python
'default_trade_amount': 0.02           # 2%
'default_stop_loss': 0.015             # 1.5%
'default_take_profit': 0.05            # 5%
'min_confidence_threshold': 0.65       # Higher confidence
```

### Ethereum (ETH)
```python
'default_trade_amount': 0.025          # 2.5%
'default_stop_loss': 0.018             # 1.8%
'default_take_profit': 0.06            # 6%
'min_confidence_threshold': 0.62       # Moderate confidence
```

### Altcoins (High Volatility)
```python
'default_trade_amount': 0.015          # 1.5%
'default_stop_loss': 0.025             # 2.5%
'default_take_profit': 0.08            # 8%
'min_confidence_threshold': 0.70       # Higher confidence needed
```

## 📈 Timeframe Considerations

### 1m - 15m (Scalping)
```python
'default_take_profit': 0.01-0.02      # 1-2%
'default_stop_loss': 0.005-0.01       # 0.5-1%
'training_interval_hours': 2           # Retrain frequently
```

### 1h - 4h (Day Trading)
```python
'default_take_profit': 0.04-0.08      # 4-8%
'default_stop_loss': 0.015-0.02       # 1.5-2%
'training_interval_hours': 12          # Standard retraining
```

### 1d (Swing Trading)
```python
'default_take_profit': 0.10-0.20      # 10-20%
'default_stop_loss': 0.03-0.05        # 3-5%
'training_interval_hours': 24          # Daily retraining
```

## ⚠️ Important Notes

1. **Backtesting**: Always test parameters in demo mode first
2. **Market Conditions**: Adjust parameters based on current market regime
3. **Position Size**: Never risk more than you can afford to lose
4. **Diversification**: Consider trading multiple timeframes or assets
5. **Monitoring**: Regularly review performance metrics and adjust
6. **Retraining**: More frequent in volatile markets, less in stable ones

## 🔍 Monitoring Checklist

Daily:
- [ ] Check win rate (should be >50%)
- [ ] Review open positions and P/L
- [ ] Monitor ML confidence levels
- [ ] Check for any errors in logs

Weekly:
- [ ] Calculate Sharpe ratio
- [ ] Review max drawdown
- [ ] Analyze profit factor
- [ ] Adjust parameters if needed

Monthly:
- [ ] Full performance review
- [ ] Compare different timeframes
- [ ] Optimize model hyperparameters
- [ ] Update strategy based on learnings
