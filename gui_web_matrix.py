"""
Matrix-Themed Web GUI for Crypto Trading Bot
============================================
Modern, stylish web application similar to Riot Launcher or Spotify
with Matrix-style visual design (green/black theme).

This creates a local web server that can be accessed through a browser.

Features:
- Dark Matrix theme with green accents
- Side navigation panel
- Real-time data visualization
- Animated effects and transitions
- Modern UI components
- Professional dashboard layout
- Works in any modern web browser

Author: BeoWulf5011
Version: 2.0.0
"""

import http.server
import socketserver
import json
import threading
import webbrowser
import os
from urllib.parse import urlparse, parse_qs

# Import the trading bot
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

PORT = 8080

# HTML template for the Matrix-themed GUI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CryptoBot Matrix Edition</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --bg-dark: #0a0e0f;
            --bg-medium: #0d1117;
            --bg-light: #161b22;
            --accent-green: #00ff41;
            --accent-dark-green: #00b530;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --text-matrix: #00ff41;
            --border: #30363d;
            --error: #f85149;
            --success: #3fb950;
            --warning: #d29922;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-primary);
            overflow: hidden;
            height: 100vh;
        }

        #app-container {
            display: flex;
            height: 100vh;
        }

        /* Sidebar */
        #sidebar {
            width: 250px;
            background-color: var(--bg-medium);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
        }

        #logo {
            padding: 30px 20px;
            text-align: center;
            border-bottom: 1px solid var(--border);
        }

        #logo h1 {
            font-size: 24px;
            color: var(--accent-green);
            margin-bottom: 5px;
            text-shadow: 0 0 10px var(--accent-green);
        }

        #logo p {
            font-size: 10px;
            color: var(--text-matrix);
            letter-spacing: 2px;
        }

        #nav {
            flex: 1;
            padding: 20px 10px;
        }

        .nav-button {
            display: block;
            width: 100%;
            padding: 15px 20px;
            margin-bottom: 5px;
            background-color: var(--bg-light);
            color: var(--text-primary);
            border: none;
            border-radius: 5px;
            text-align: left;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .nav-button:hover {
            background-color: var(--accent-dark-green);
            color: var(--bg-dark);
            transform: translateX(5px);
        }

        .nav-button.active {
            background-color: var(--accent-green);
            color: var(--bg-dark);
        }

        #status {
            padding: 20px;
            background-color: var(--bg-light);
            border-top: 1px solid var(--border);
            text-align: center;
        }

        #status h3 {
            font-size: 10px;
            color: var(--text-secondary);
            margin-bottom: 5px;
        }

        #status p {
            color: var(--success);
            font-weight: bold;
        }

        /* Main content */
        #main-content {
            flex: 1;
            overflow-y: auto;
            padding: 40px;
        }

        #main-content::-webkit-scrollbar {
            width: 10px;
        }

        #main-content::-webkit-scrollbar-track {
            background: var(--bg-dark);
        }

        #main-content::-webkit-scrollbar-thumb {
            background: var(--accent-dark-green);
            border-radius: 5px;
        }

        .header {
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 32px;
            color: var(--text-matrix);
            margin-bottom: 10px;
            text-shadow: 0 0 10px var(--accent-green);
        }

        .header p {
            color: var(--text-secondary);
            font-size: 14px;
        }

        .separator {
            height: 1px;
            background-color: var(--border);
            margin: 20px 0;
        }

        /* Cards */
        .card {
            background-color: var(--bg-light);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .card-header {
            font-size: 18px;
            font-weight: bold;
            color: var(--text-primary);
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border);
        }

        .card-content {
            color: var(--text-secondary);
        }

        /* Stats grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-card {
            background: linear-gradient(135deg, var(--bg-light) 0%, var(--bg-medium) 100%);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 25px;
            text-align: center;
            transition: all 0.3s ease;
        }

        .stat-card:hover {
            transform: translateY(-5px);
            border-color: var(--accent-green);
            box-shadow: 0 5px 20px rgba(0, 255, 65, 0.2);
        }

        .stat-icon {
            font-size: 32px;
            margin-bottom: 10px;
        }

        .stat-value {
            font-size: 28px;
            font-weight: bold;
            color: var(--text-matrix);
            margin-bottom: 5px;
        }

        .stat-label {
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Form elements */
        .form-group {
            margin-bottom: 20px;
        }

        .form-label {
            display: block;
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .form-input, .form-select {
            width: 100%;
            padding: 12px 15px;
            background-color: var(--bg-medium);
            border: 1px solid var(--border);
            border-radius: 5px;
            color: var(--text-primary);
            font-size: 14px;
            transition: all 0.3s ease;
        }

        .form-input:focus, .form-select:focus {
            outline: none;
            border-color: var(--accent-green);
            box-shadow: 0 0 10px rgba(0, 255, 65, 0.3);
        }

        /* Buttons */
        .btn {
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-right: 10px;
        }

        .btn-primary {
            background-color: var(--accent-green);
            color: var(--bg-dark);
        }

        .btn-primary:hover {
            background-color: var(--accent-dark-green);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 255, 65, 0.4);
        }

        .btn-secondary {
            background-color: var(--bg-light);
            color: var(--text-primary);
            border: 1px solid var(--border);
        }

        .btn-secondary:hover {
            background-color: var(--bg-medium);
            border-color: var(--accent-green);
        }

        .btn-danger {
            background-color: var(--error);
            color: var(--text-primary);
        }

        .btn-danger:hover {
            background-color: #d04040;
        }

        /* Activity list */
        .activity-item {
            display: flex;
            align-items: center;
            padding: 15px 0;
            border-bottom: 1px solid var(--border);
        }

        .activity-item:last-child {
            border-bottom: none;
        }

        .activity-dot {
            width: 10px;
            height: 10px;
            background-color: var(--accent-green);
            border-radius: 50%;
            margin-right: 15px;
        }

        .activity-content {
            flex: 1;
        }

        .activity-title {
            color: var(--text-primary);
            font-weight: bold;
            margin-bottom: 5px;
        }

        .activity-desc {
            color: var(--text-secondary);
            font-size: 12px;
        }

        .activity-time {
            color: var(--text-secondary);
            font-size: 12px;
        }

        /* Info grid */
        .info-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }

        .info-item {
            padding: 15px 0;
        }

        .info-label {
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 5px;
        }

        .info-value {
            font-size: 20px;
            font-weight: bold;
            color: var(--text-matrix);
        }

        /* Matrix rain animation */
        #matrix-canvas {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            opacity: 0.1;
        }

        /* Loading animation */
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid var(--border);
            border-radius: 50%;
            border-top-color: var(--accent-green);
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Hidden by default */
        .view {
            display: none;
        }

        .view.active {
            display: block;
        }

        /* Responsive */
        @media (max-width: 768px) {
            #sidebar {
                width: 200px;
            }

            #main-content {
                padding: 20px;
            }

            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <!-- Matrix rain effect canvas -->
    <canvas id="matrix-canvas"></canvas>

    <div id="app-container">
        <!-- Sidebar -->
        <div id="sidebar">
            <div id="logo">
                <h1>⚡ CRYPTOBOT</h1>
                <p>MATRIX EDITION</p>
            </div>

            <div id="nav">
                <button class="nav-button active" onclick="showView('dashboard')">📊 Dashboard</button>
                <button class="nav-button" onclick="showView('config')">⚙️ Configuration</button>
                <button class="nav-button" onclick="showView('monitor')">📈 Trading Monitor</button>
                <button class="nav-button" onclick="showView('charts')">📉 Charts & Analysis</button>
                <button class="nav-button" onclick="showView('system')">💻 System Info</button>
                <button class="nav-button" onclick="showView('history')">📜 Trade History</button>
            </div>

            <div id="status">
                <h3>SYSTEM STATUS</h3>
                <p>● ONLINE</p>
            </div>
        </div>

        <!-- Main content -->
        <div id="main-content">
            <!-- Dashboard View -->
            <div id="dashboard-view" class="view active">
                <div class="header">
                    <h1>Dashboard</h1>
                    <p>Real-time trading bot overview</p>
                </div>
                <div class="separator"></div>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-icon">💰</div>
                        <div class="stat-value">$0.00</div>
                        <div class="stat-label">Total Balance</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon">📈</div>
                        <div class="stat-value">$0.00</div>
                        <div class="stat-label">Total Profit</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon">🎯</div>
                        <div class="stat-value">0%</div>
                        <div class="stat-label">Win Rate</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon">🤖</div>
                        <div class="stat-value">0</div>
                        <div class="stat-label">Active Bots</div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">Recent Activity</div>
                    <div class="card-content">
                        <div class="activity-item">
                            <div class="activity-dot"></div>
                            <div class="activity-content">
                                <div class="activity-title">System Initialized</div>
                                <div class="activity-desc">System started successfully</div>
                            </div>
                            <div class="activity-time">2 min ago</div>
                        </div>
                        <div class="activity-item">
                            <div class="activity-dot"></div>
                            <div class="activity-content">
                                <div class="activity-title">Waiting for Configuration</div>
                                <div class="activity-desc">Configure bot to start trading</div>
                            </div>
                            <div class="activity-time">now</div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">Quick Actions</div>
                    <div class="card-content">
                        <button class="btn btn-primary" onclick="showView('config')">⚙️ Configure Bot</button>
                        <button class="btn btn-secondary" onclick="showView('charts')">📊 View Charts</button>
                    </div>
                </div>
            </div>

            <!-- Configuration View -->
            <div id="config-view" class="view">
                <div class="header">
                    <h1>Configuration</h1>
                    <p>Configure your trading bot settings</p>
                </div>
                <div class="separator"></div>

                <div class="card">
                    <div class="card-header">Exchange Settings</div>
                    <div class="card-content">
                        <div class="form-group">
                            <label class="form-label">Exchange</label>
                            <select class="form-select" id="exchange">
                                <option>binance</option>
                                <option>coinbase</option>
                                <option>kraken</option>
                                <option>kucoin</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">API Key</label>
                            <input type="password" class="form-input" id="api-key" placeholder="Enter API Key">
                        </div>
                        <div class="form-group">
                            <label class="form-label">API Secret</label>
                            <input type="password" class="form-input" id="api-secret" placeholder="Enter API Secret">
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">Trading Settings</div>
                    <div class="card-content">
                        <div class="form-group">
                            <label class="form-label">Trading Pair</label>
                            <select class="form-select" id="symbol">
                                <option>BTC/USDT</option>
                                <option>ETH/USDT</option>
                                <option>SOL/USDT</option>
                                <option>XRP/USDT</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Timeframe</label>
                            <select class="form-select" id="timeframe">
                                <option>1m</option>
                                <option>5m</option>
                                <option>15m</option>
                                <option>30m</option>
                                <option selected>1h</option>
                                <option>4h</option>
                                <option>1d</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Trade Size (%)</label>
                            <input type="number" class="form-input" id="trade-size" value="2.0" step="0.1">
                        </div>
                        <div class="form-group">
                            <label class="form-label">Stop Loss (%)</label>
                            <input type="number" class="form-input" id="stop-loss" value="2.0" step="0.1">
                        </div>
                        <div class="form-group">
                            <label class="form-label">Take Profit (%)</label>
                            <input type="number" class="form-input" id="take-profit" value="5.0" step="0.1">
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">AI/ML Settings</div>
                    <div class="card-content">
                        <div class="form-group">
                            <label class="form-label">Training Interval (hours)</label>
                            <input type="number" class="form-input" id="train-interval" value="12" min="1">
                        </div>
                        <div class="form-group">
                            <label class="form-label">
                                <input type="checkbox" checked> Enable GPU Acceleration (RTX 3080)
                            </label>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-content">
                        <button class="btn btn-primary" onclick="startBot()">▶ Start Trading Bot</button>
                        <button class="btn btn-danger" onclick="stopBot()">⏹ Stop Bot</button>
                        <button class="btn btn-secondary" onclick="testConnection()">🔌 Test Connection</button>
                    </div>
                </div>
            </div>

            <!-- Monitor View -->
            <div id="monitor-view" class="view">
                <div class="header">
                    <h1>Trading Monitor</h1>
                    <p>Live market data and bot activity</p>
                </div>
                <div class="separator"></div>

                <div class="card">
                    <div class="card-header">Market Information</div>
                    <div class="card-content">
                        <div class="info-grid">
                            <div class="info-item">
                                <div class="info-label">Current Price</div>
                                <div class="info-value">$--</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Predicted Price</div>
                                <div class="info-value">$--</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Market Sentiment</div>
                                <div class="info-value">--</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Trend</div>
                                <div class="info-value">--</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Balance</div>
                                <div class="info-value">$--</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Total P&L</div>
                                <div class="info-value">$--</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">Recent Trades</div>
                    <div class="card-content">
                        <p style="text-align: center; padding: 50px; color: var(--text-secondary);">
                            No trades yet. Start the bot to begin trading.
                        </p>
                    </div>
                </div>

                <div class="card">
                    <div class="card-content">
                        <button class="btn btn-secondary" onclick="refreshData()">🔄 Refresh Data</button>
                    </div>
                </div>
            </div>

            <!-- Charts View -->
            <div id="charts-view" class="view">
                <div class="header">
                    <h1>Charts & Analysis</h1>
                    <p>Visual analysis and predictions</p>
                </div>
                <div class="separator"></div>

                <div class="card">
                    <div class="card-content">
                        <button class="btn btn-primary" onclick="generateChart()">📊 Generate Price Chart</button>
                        <button class="btn btn-secondary" onclick="generateDashboard()">📈 Performance Dashboard</button>
                        <button class="btn btn-secondary" onclick="exportHistory()">💾 Export History</button>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">Chart Visualization</div>
                    <div class="card-content">
                        <p style="text-align: center; padding: 100px; color: var(--text-secondary); font-size: 24px;">
                            📊<br><br>
                            Generate a chart to display here
                        </p>
                    </div>
                </div>
            </div>

            <!-- System View -->
            <div id="system-view" class="view">
                <div class="header">
                    <h1>System Information</h1>
                    <p>Hardware and software status</p>
                </div>
                <div class="separator"></div>

                <div class="card">
                    <div class="card-header">Hardware Information</div>
                    <div class="card-content">
                        <div class="info-grid">
                            <div class="info-item">
                                <div class="info-label">💻 CPU</div>
                                <div class="info-value" style="font-size: 16px;">Intel i9-13900K</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">🎮 GPU</div>
                                <div class="info-value" style="font-size: 16px;">NVIDIA RTX 3080</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">💾 RAM</div>
                                <div class="info-value" style="font-size: 16px;">32 GB DDR5</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">⚡ CUDA</div>
                                <div class="info-value" style="font-size: 16px;">Enabled</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">Software Versions</div>
                    <div class="card-content">
                        <div class="info-grid">
                            <div class="info-item">
                                <div class="info-label">🐍 Python</div>
                                <div class="info-value" style="font-size: 16px;">3.10+</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">🧠 TensorFlow</div>
                                <div class="info-value" style="font-size: 16px;">2.x</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">📦 NumPy</div>
                                <div class="info-value" style="font-size: 16px;">Latest</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">📊 Pandas</div>
                                <div class="info-value" style="font-size: 16px;">Latest</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">Resource Monitoring</div>
                    <div class="card-content">
                        <p style="text-align: center; padding: 30px; color: var(--text-secondary);">
                            Real-time resource monitoring<br>
                            CPU • GPU • Memory • Temperature
                        </p>
                        <div style="text-align: center;">
                            <button class="btn btn-secondary" onclick="refreshSystem()">🔄 Refresh</button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- History View -->
            <div id="history-view" class="view">
                <div class="header">
                    <h1>Trade History</h1>
                    <p>View and export your trading history</p>
                </div>
                <div class="separator"></div>

                <div class="card">
                    <div class="card-header">Trading History</div>
                    <div class="card-content">
                        <p style="text-align: center; padding: 100px; color: var(--text-secondary); font-size: 20px;">
                            📜<br><br>
                            No trading history available yet.<br>
                            Start trading to see your history here.
                        </p>
                    </div>
                </div>

                <div class="card">
                    <div class="card-content">
                        <button class="btn btn-secondary" onclick="exportJSON()">💾 Export as JSON</button>
                        <button class="btn btn-secondary" onclick="exportCSV()">📊 Export as CSV</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Matrix rain effect
        const canvas = document.getElementById('matrix-canvas');
        const ctx = canvas.getContext('2d');

        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        const matrix = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789@#$%^&*()*&^%+-/~{[|`]}";
        const fontSize = 16;
        const columns = canvas.width / fontSize;

        const drops = [];
        for (let i = 0; i < columns; i++) {
            drops[i] = 1;
        }

        function drawMatrix() {
            ctx.fillStyle = 'rgba(10, 14, 15, 0.05)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.fillStyle = '#00ff41';
            ctx.font = fontSize + 'px monospace';

            for (let i = 0; i < drops.length; i++) {
                const text = matrix[Math.floor(Math.random() * matrix.length)];
                ctx.fillText(text, i * fontSize, drops[i] * fontSize);

                if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
                    drops[i] = 0;
                }
                drops[i]++;
            }
        }

        setInterval(drawMatrix, 35);

        window.addEventListener('resize', () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        });

        // View switching
        function showView(viewName) {
            // Hide all views
            document.querySelectorAll('.view').forEach(view => {
                view.classList.remove('active');
            });

            // Show selected view
            document.getElementById(viewName + '-view').classList.add('active');

            // Update nav buttons
            document.querySelectorAll('.nav-button').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
        }

        // Bot functions
        function startBot() {
            alert('Trading bot will be started with configured settings.\\n\\nThis is a demo - full integration available in the Python backend.');
        }

        function stopBot() {
            alert('Trading bot will be stopped.');
        }

        function testConnection() {
            alert('Testing connection to exchange...\\n\\nThis is a demo - full integration available in the Python backend.');
        }

        function refreshData() {
            alert('Data refreshed');
        }

        function generateChart() {
            alert('Generating price prediction chart...\\n\\nCharts will be displayed here with full backend integration.');
        }

        function generateDashboard() {
            alert('Generating performance dashboard...');
        }

        function exportHistory() {
            alert('Exporting trading history...');
        }

        function refreshSystem() {
            alert('System information refreshed');
        }

        function exportJSON() {
            alert('Exporting trading history as JSON...');
        }

        function exportCSV() {
            alert('Exporting trading history as CSV...');
        }

        // Initialize
        console.log('CryptoBot Matrix Edition initialized');
        console.log('Version 2.0.0');
    </script>
</body>
</html>
"""


class MatrixGUIHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler for the Matrix GUI."""
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
        else:
            super().do_GET()
    
    def log_message(self, format, *args):
        """Override to reduce console output."""
        pass


def launch_web_gui(port=PORT, auto_open=True):
    """Launch the web-based Matrix GUI."""
    print("=" * 80)
    print(" CryptoBot Matrix Edition - Web GUI")
    print("=" * 80)
    print(f" Starting web server on http://localhost:{port}")
    print(" Press Ctrl+C to stop the server")
    print("=" * 80)
    print()
    
    # Create server
    with socketserver.TCPServer(("", port), MatrixGUIHandler) as httpd:
        url = f"http://localhost:{port}"
        
        if auto_open:
            # Open browser in a separate thread
            def open_browser():
                import time
                time.sleep(1)  # Wait for server to start
                webbrowser.open(url)
            
            browser_thread = threading.Thread(target=open_browser)
            browser_thread.daemon = True
            browser_thread.start()
        
        print(f"✓ Server started successfully!")
        print(f"✓ Open your browser and navigate to: {url}")
        print()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nShutting down server...")
            httpd.shutdown()
            print("Server stopped.")


if __name__ == "__main__":
    launch_web_gui()
