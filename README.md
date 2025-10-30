# CryptoBot Matrix Edition 🚀

## Version 2.0.0 - Matrix-Themed GUI

A high-performance cryptocurrency trading bot with AI/ML capabilities and a stunning Matrix-themed user interface inspired by modern applications like Riot Launcher and Spotify.

## ✨ Features

### Core Trading Features
- **Deep Learning Prediction** - LSTM neural networks with GPU acceleration (NVIDIA RTX 3080)
- **Advanced Technical Indicators** - 50+ indicators including MACD, RSI, Bollinger Bands, ADX, Ichimoku
- **Market Sentiment Analysis** - AI-powered sentiment analysis with machine learning
- **Risk Management** - Adaptive position sizing, stop-loss, and take-profit
- **Multi-Exchange Support** - Binance, Coinbase, Kraken, KuCoin
- **Real-time Data** - Live market data and predictions

### Hardware Optimization
- **NVIDIA RTX 3080** - CUDA acceleration for neural network training
- **Intel i9-13900K** - Multi-threading support for parallel processing
- **32GB RAM** - Efficient memory management
- **Mixed Precision Training** - FP16 for faster GPU computations

### 🎨 Matrix-Themed GUI

The new GUI brings a professional, modern interface with:

#### Design
- **Dark Matrix Theme** - Sleek black background with neon green accents
- **Matrix Rain Effect** - Animated background with falling characters
- **Modern Typography** - Clean, readable fonts
- **Smooth Animations** - Hover effects and transitions
- **Responsive Layout** - Works on different screen sizes

#### Navigation
- **📊 Dashboard** - Overview of balance, profit, win rate, and active bots
- **⚙️ Configuration** - Easy setup of exchange, trading pairs, and AI settings
- **📈 Trading Monitor** - Real-time market data and bot activity
- **📉 Charts & Analysis** - Visual price predictions and performance metrics
- **💻 System Info** - Hardware and software status monitoring
- **📜 Trade History** - Complete history of all trades

#### User Experience
- **Web-Based** - Accessible through any modern web browser
- **No Installation** - Just run and open in browser
- **Cross-Platform** - Works on Windows, Linux, macOS
- **Professional Look** - Similar to Riot Launcher, Spotify, VS Code

## 🚀 Quick Start

### Running the Bot

#### 1. Web GUI (Recommended)
```bash
python3 crypto-trading-bot.py --gui
```
This will automatically launch the Matrix-themed web interface at `http://localhost:8080`

#### 2. Command Line Interface
```bash
python3 crypto-trading-bot.py --symbol BTC/USDT --exchange binance --timeframe 1h
```

### Configuration

1. Open the **Configuration** tab in the GUI
2. Enter your exchange API credentials
3. Select trading pair and timeframe
4. Set risk parameters (trade size, stop loss, take profit)
5. Configure AI/ML settings
6. Click "Start Trading Bot"

## 📋 Requirements

### Python Packages
```bash
pip install numpy pandas tensorflow ccxt scikit-learn matplotlib
```

### Optional (for enhanced features)
```bash
pip install pynvml psutil pillow
```

## 🎯 Trading Strategy

The bot uses a sophisticated multi-factor strategy:

1. **Technical Analysis** - Analyzes 50+ indicators across multiple timeframes
2. **AI Predictions** - LSTM neural network predicts future price movements
3. **Sentiment Scoring** - Weighted sentiment analysis combining all signals
4. **Risk Management** - Dynamic position sizing based on volatility
5. **Trade Execution** - Automated buy/sell with stop-loss and take-profit

## 📊 Performance Tracking

- **Real-time Monitoring** - Live P&L, win rate, and trade statistics
- **Performance Reports** - Detailed JSON reports saved automatically
- **Visual Charts** - Price prediction charts and performance dashboards
- **Trade History** - Complete log of all trades with export capability

## 🔒 Security

- **API Keys** - Stored securely, never logged
- **Read-Only Mode** - Option to run without trading permissions
- **Demo Mode** - Test strategies without real money
- **Risk Limits** - Maximum position size and loss limits

## 💡 Tips for Best Results

1. **Start Small** - Begin with small position sizes (1-2%)
2. **Use Stop-Loss** - Always set stop-loss to protect capital
3. **Monitor Performance** - Regularly check the dashboard
4. **Retrain Model** - Retrain the AI model every 12-24 hours
5. **Diversify** - Run multiple bots on different pairs
6. **Stay Updated** - Keep software and exchange APIs up to date

## 🎨 GUI Color Scheme

The Matrix theme uses:
- **Background**: Dark (#0a0e0f, #0d1117, #161b22)
- **Accent**: Matrix Green (#00ff41, #00b530)
- **Text**: Light Gray (#c9d1d9, #8b949e)
- **Borders**: Dark Gray (#30363d)
- **Status Colors**: Green (success), Red (error), Yellow (warning), Blue (info)

## 📝 File Structure

```
├── crypto-trading-bot.py       # Main trading bot engine
├── gui_web_matrix.py          # Matrix-themed web GUI
├── gui_matrix_theme.py        # Matrix-themed Tkinter GUI (fallback)
├── README.md                  # This file
├── trading_bot.log            # Bot activity log
└── data/                      # Data directory
    ├── models/                # Saved AI models
    ├── reports/               # Performance reports
    ├── visualizations/        # Generated charts
    └── trades/                # Trade history
```

## 🐛 Troubleshooting

### GUI Won't Start
- Ensure no other process is using port 8080
- Try a different port: `python3 gui_web_matrix.py --port 8081`

### GPU Not Detected
- Install NVIDIA drivers and CUDA toolkit
- Verify with: `nvidia-smi`
- Use `--no-gpu` flag to run on CPU

### Exchange Connection Failed
- Check API credentials
- Verify network connection
- Check exchange status
- Ensure API has required permissions

### Model Training Slow
- Enable GPU acceleration
- Reduce batch size
- Use shorter lookback period
- Train less frequently

## 🔧 Advanced Configuration

### Environment Variables
```bash
export TRADING_BOT_API_KEY="your-api-key"
export TRADING_BOT_API_SECRET="your-api-secret"
export CUDA_VISIBLE_DEVICES="0"  # GPU device ID
```

### Custom Settings
Edit the configuration section in `crypto-trading-bot.py`:
```python
TRADING_PARAMS = {
    'default_trade_amount': 0.02,
    'default_stop_loss': 0.02,
    'default_take_profit': 0.05,
}
```

## 📈 Future Enhancements

- [ ] WebSocket real-time updates
- [ ] Portfolio management across multiple exchanges
- [ ] Social trading features
- [ ] Advanced charting with TradingView integration
- [ ] Backtesting framework
- [ ] Mobile responsive design
- [ ] Dark/Light theme toggle
- [ ] More AI models (Transformer, GRU)

## 🙏 Credits

- **Author**: BeoWulf5011
- **Version**: 2.0.0
- **License**: MIT
- **Hardware**: Optimized for NVIDIA RTX 3080 & Intel i9-13900K

## ⚠️ Disclaimer

Trading cryptocurrencies carries significant risk. This bot is provided for educational purposes. Always:
- Trade responsibly
- Never invest more than you can afford to lose
- Test thoroughly with demo accounts first
- Understand the risks involved
- Comply with local regulations

**Past performance does not guarantee future results.**

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the logs in `trading_bot.log`
- Review the documentation

---

**Made with ❤️ and powered by AI**

*The Matrix has you... but now you have the Matrix-themed trading bot! 🟢*
