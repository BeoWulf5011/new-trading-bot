"""
Matrix-Themed GUI Application for Crypto Trading Bot
=====================================================
Modern, stylish Windows application similar to Riot Launcher or Spotify
with Matrix-style visual design (green/black theme).

Features:
- Dark Matrix theme with green accents
- Side navigation panel
- Real-time data visualization
- Animated effects and transitions
- Modern window controls
- Professional dashboard layout

Author: BeoWulf5011
Version: 2.0.0
"""

import tkinter as tk
from tkinter import ttk, messagebox, font
import threading
import datetime
import json
import os
from typing import Dict, Any, Optional

# Color scheme - Matrix theme
COLORS = {
    'bg_dark': '#0a0e0f',          # Very dark background
    'bg_medium': '#0d1117',         # Medium dark background
    'bg_light': '#161b22',          # Lighter dark background
    'accent_green': '#00ff41',      # Matrix green
    'accent_dark_green': '#00b530', # Darker green
    'text_primary': '#c9d1d9',      # Light text
    'text_secondary': '#8b949e',    # Secondary text
    'text_matrix': '#00ff41',       # Matrix green text
    'border': '#30363d',            # Border color
    'error': '#f85149',             # Error red
    'success': '#3fb950',           # Success green
    'warning': '#d29922',           # Warning yellow
    'info': '#58a6ff',              # Info blue
}

# Typography
FONTS = {
    'title': ('Segoe UI', 24, 'bold'),
    'subtitle': ('Segoe UI', 18, 'bold'),
    'heading': ('Segoe UI', 14, 'bold'),
    'body': ('Segoe UI', 11),
    'small': ('Segoe UI', 9),
    'mono': ('Consolas', 10),
    'matrix': ('Courier New', 10, 'bold'),
}


class MatrixWindow(tk.Tk):
    """Custom window with Matrix theme and modern styling."""
    
    def __init__(self):
        super().__init__()
        
        # Window configuration
        self.title("CryptoBot Matrix")
        self.geometry("1600x900")
        self.minsize(1200, 700)
        
        # Remove default window decorations (we'll create custom ones)
        self.overrideredirect(False)  # Keep OS decorations for now
        
        # Configure window
        self.configure(bg=COLORS['bg_dark'])
        
        # Window state
        self.is_maximized = False
        
        # Create main container
        self.main_container = tk.Frame(self, bg=COLORS['bg_dark'])
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Initialize components
        self.sidebar = None
        self.content_frame = None
        self.current_view = None
        
        # Bot management
        self.bots = {}
        self.active_threads = {}
        
        # Build the interface
        self.build_ui()
        
    def build_ui(self):
        """Build the main user interface."""
        # Create sidebar
        self.create_sidebar()
        
        # Create content area
        self.create_content_area()
        
        # Load default view (Dashboard)
        self.show_dashboard()
        
    def create_sidebar(self):
        """Create the side navigation panel."""
        self.sidebar = tk.Frame(
            self.main_container,
            bg=COLORS['bg_medium'],
            width=250
        )
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)
        
        # Logo/Title section
        logo_frame = tk.Frame(self.sidebar, bg=COLORS['bg_medium'], height=100)
        logo_frame.pack(fill=tk.X, pady=20, padx=15)
        
        title_label = tk.Label(
            logo_frame,
            text="⚡ CRYPTOBOT",
            font=FONTS['title'],
            bg=COLORS['bg_medium'],
            fg=COLORS['accent_green']
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            logo_frame,
            text="MATRIX EDITION",
            font=FONTS['small'],
            bg=COLORS['bg_medium'],
            fg=COLORS['text_matrix']
        )
        subtitle_label.pack()
        
        # Separator
        separator = tk.Frame(self.sidebar, bg=COLORS['border'], height=1)
        separator.pack(fill=tk.X, padx=15, pady=10)
        
        # Navigation buttons
        nav_buttons = [
            ("📊 Dashboard", self.show_dashboard),
            ("⚙️ Configuration", self.show_configuration),
            ("📈 Trading Monitor", self.show_monitor),
            ("📉 Charts & Analysis", self.show_charts),
            ("💻 System Info", self.show_system),
            ("📜 Trade History", self.show_history),
        ]
        
        self.nav_button_widgets = {}
        
        for text, command in nav_buttons:
            btn = self.create_nav_button(text, command)
            self.nav_button_widgets[text] = btn
        
        # Spacer
        tk.Frame(self.sidebar, bg=COLORS['bg_medium']).pack(fill=tk.BOTH, expand=True)
        
        # Status section at bottom
        status_frame = tk.Frame(self.sidebar, bg=COLORS['bg_light'], height=80)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        
        tk.Label(
            status_frame,
            text="System Status",
            font=FONTS['small'],
            bg=COLORS['bg_light'],
            fg=COLORS['text_secondary']
        ).pack(pady=(10, 5))
        
        self.status_indicator = tk.Label(
            status_frame,
            text="● ONLINE",
            font=FONTS['body'],
            bg=COLORS['bg_light'],
            fg=COLORS['success']
        )
        self.status_indicator.pack()
        
    def create_nav_button(self, text: str, command):
        """Create a styled navigation button."""
        btn_frame = tk.Frame(self.sidebar, bg=COLORS['bg_medium'])
        btn_frame.pack(fill=tk.X, padx=10, pady=3)
        
        btn = tk.Button(
            btn_frame,
            text=text,
            font=FONTS['body'],
            bg=COLORS['bg_light'],
            fg=COLORS['text_primary'],
            activebackground=COLORS['accent_dark_green'],
            activeforeground=COLORS['bg_dark'],
            relief=tk.FLAT,
            anchor='w',
            padx=20,
            pady=12,
            cursor='hand2',
            command=command,
            borderwidth=0,
            highlightthickness=0
        )
        btn.pack(fill=tk.X)
        
        # Hover effects
        def on_enter(e):
            btn.configure(bg=COLORS['accent_dark_green'], fg=COLORS['bg_dark'])
            
        def on_leave(e):
            btn.configure(bg=COLORS['bg_light'], fg=COLORS['text_primary'])
        
        btn.bind('<Enter>', on_enter)
        btn.bind('<Leave>', on_leave)
        
        return btn
    
    def create_content_area(self):
        """Create the main content area."""
        self.content_frame = tk.Frame(
            self.main_container,
            bg=COLORS['bg_dark']
        )
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
    def clear_content(self):
        """Clear the content area."""
        if self.current_view:
            self.current_view.destroy()
            self.current_view = None
    
    def show_dashboard(self):
        """Show the dashboard view."""
        self.clear_content()
        self.current_view = DashboardView(self.content_frame, self)
        
    def show_configuration(self):
        """Show the configuration view."""
        self.clear_content()
        self.current_view = ConfigurationView(self.content_frame, self)
        
    def show_monitor(self):
        """Show the trading monitor view."""
        self.clear_content()
        self.current_view = MonitorView(self.content_frame, self)
        
    def show_charts(self):
        """Show the charts and analysis view."""
        self.clear_content()
        self.current_view = ChartsView(self.content_frame, self)
        
    def show_system(self):
        """Show the system information view."""
        self.clear_content()
        self.current_view = SystemView(self.content_frame, self)
        
    def show_history(self):
        """Show the trade history view."""
        self.clear_content()
        self.current_view = HistoryView(self.content_frame, self)


class BaseView(tk.Frame):
    """Base class for all views."""
    
    def __init__(self, parent, app):
        super().__init__(parent, bg=COLORS['bg_dark'])
        self.app = app
        self.pack(fill=tk.BOTH, expand=True)
        self.create_widgets()
        
    def create_widgets(self):
        """Override this method in subclasses."""
        pass
    
    def create_header(self, title: str, subtitle: str = ""):
        """Create a header section."""
        header_frame = tk.Frame(self, bg=COLORS['bg_dark'])
        header_frame.pack(fill=tk.X, padx=30, pady=(30, 10))
        
        tk.Label(
            header_frame,
            text=title,
            font=FONTS['title'],
            bg=COLORS['bg_dark'],
            fg=COLORS['text_matrix']
        ).pack(anchor='w')
        
        if subtitle:
            tk.Label(
                header_frame,
                text=subtitle,
                font=FONTS['body'],
                bg=COLORS['bg_dark'],
                fg=COLORS['text_secondary']
            ).pack(anchor='w', pady=(5, 0))
        
        # Separator line
        tk.Frame(
            self,
            bg=COLORS['border'],
            height=1
        ).pack(fill=tk.X, padx=30, pady=15)
        
    def create_card(self, parent, title: str = "", height: int = None):
        """Create a styled card container."""
        card = tk.Frame(
            parent,
            bg=COLORS['bg_light'],
            highlightbackground=COLORS['border'],
            highlightthickness=1
        )
        
        if title:
            title_frame = tk.Frame(card, bg=COLORS['bg_medium'], height=40)
            title_frame.pack(fill=tk.X)
            title_frame.pack_propagate(False)
            
            tk.Label(
                title_frame,
                text=title,
                font=FONTS['heading'],
                bg=COLORS['bg_medium'],
                fg=COLORS['text_primary']
            ).pack(side=tk.LEFT, padx=15, pady=10)
        
        content = tk.Frame(card, bg=COLORS['bg_light'])
        content.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        if height:
            card.configure(height=height)
            card.pack_propagate(False)
        
        return card, content
    
    def create_button(self, parent, text: str, command, style='primary'):
        """Create a styled button."""
        colors_map = {
            'primary': (COLORS['accent_green'], COLORS['bg_dark']),
            'secondary': (COLORS['bg_light'], COLORS['text_primary']),
            'danger': (COLORS['error'], COLORS['text_primary']),
        }
        
        bg_color, fg_color = colors_map.get(style, colors_map['primary'])
        
        btn = tk.Button(
            parent,
            text=text,
            font=FONTS['body'],
            bg=bg_color,
            fg=fg_color,
            activebackground=COLORS['accent_dark_green'],
            activeforeground=COLORS['bg_dark'],
            relief=tk.FLAT,
            padx=25,
            pady=10,
            cursor='hand2',
            command=command,
            borderwidth=0
        )
        
        return btn
    
    def create_input(self, parent, label: str, default: str = "", show: str = None):
        """Create a labeled input field."""
        frame = tk.Frame(parent, bg=COLORS['bg_light'])
        
        tk.Label(
            frame,
            text=label,
            font=FONTS['body'],
            bg=COLORS['bg_light'],
            fg=COLORS['text_secondary']
        ).pack(anchor='w', pady=(0, 5))
        
        entry = tk.Entry(
            frame,
            font=FONTS['body'],
            bg=COLORS['bg_medium'],
            fg=COLORS['text_primary'],
            insertbackground=COLORS['accent_green'],
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=COLORS['border'],
            highlightcolor=COLORS['accent_green'],
            show=show
        )
        entry.insert(0, default)
        entry.pack(fill=tk.X, ipady=8, ipadx=10)
        
        return frame, entry


class DashboardView(BaseView):
    """Dashboard view - main overview."""
    
    def create_widgets(self):
        self.create_header("Dashboard", "Real-time trading bot overview")
        
        # Scrollable content
        canvas = tk.Canvas(self, bg=COLORS['bg_dark'], highlightthickness=0)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=COLORS['bg_dark'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Stats cards row
        stats_frame = tk.Frame(scrollable_frame, bg=COLORS['bg_dark'])
        stats_frame.pack(fill=tk.X, padx=30, pady=10)
        
        # Create stat cards
        self.create_stat_card(stats_frame, "Total Balance", "$0.00", "💰").pack(
            side=tk.LEFT, padx=10, expand=True, fill=tk.BOTH
        )
        self.create_stat_card(stats_frame, "Total Profit", "$0.00", "📈").pack(
            side=tk.LEFT, padx=10, expand=True, fill=tk.BOTH
        )
        self.create_stat_card(stats_frame, "Win Rate", "0%", "🎯").pack(
            side=tk.LEFT, padx=10, expand=True, fill=tk.BOTH
        )
        self.create_stat_card(stats_frame, "Active Bots", "0", "🤖").pack(
            side=tk.LEFT, padx=10, expand=True, fill=tk.BOTH
        )
        
        # Recent activity
        activity_card, activity_content = self.create_card(
            scrollable_frame, "Recent Activity"
        )
        activity_card.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)
        
        # Activity list
        activities = [
            ("System Initialized", "System started successfully", "2 min ago"),
            ("Waiting for Configuration", "Configure bot to start trading", "now"),
        ]
        
        for title, desc, time in activities:
            self.create_activity_item(activity_content, title, desc, time)
        
        # Quick actions
        actions_frame = tk.Frame(scrollable_frame, bg=COLORS['bg_dark'])
        actions_frame.pack(fill=tk.X, padx=30, pady=20)
        
        tk.Label(
            actions_frame,
            text="Quick Actions",
            font=FONTS['heading'],
            bg=COLORS['bg_dark'],
            fg=COLORS['text_primary']
        ).pack(anchor='w', pady=(0, 10))
        
        btn_frame = tk.Frame(actions_frame, bg=COLORS['bg_dark'])
        btn_frame.pack(fill=tk.X)
        
        self.create_button(btn_frame, "⚙️ Configure Bot", self.app.show_configuration).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        self.create_button(btn_frame, "📊 View Charts", self.app.show_charts, 'secondary').pack(
            side=tk.LEFT, padx=(0, 10)
        )
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_stat_card(self, parent, title, value, icon):
        """Create a statistics card."""
        card = tk.Frame(
            parent,
            bg=COLORS['bg_light'],
            highlightbackground=COLORS['border'],
            highlightthickness=1
        )
        
        icon_label = tk.Label(
            card,
            text=icon,
            font=('Segoe UI Emoji', 24),
            bg=COLORS['bg_light'],
            fg=COLORS['accent_green']
        )
        icon_label.pack(pady=(15, 5))
        
        value_label = tk.Label(
            card,
            text=value,
            font=FONTS['subtitle'],
            bg=COLORS['bg_light'],
            fg=COLORS['text_matrix']
        )
        value_label.pack()
        
        title_label = tk.Label(
            card,
            text=title,
            font=FONTS['small'],
            bg=COLORS['bg_light'],
            fg=COLORS['text_secondary']
        )
        title_label.pack(pady=(5, 15))
        
        return card
    
    def create_activity_item(self, parent, title, description, time):
        """Create an activity list item."""
        item_frame = tk.Frame(parent, bg=COLORS['bg_light'])
        item_frame.pack(fill=tk.X, pady=8)
        
        # Indicator dot
        tk.Label(
            item_frame,
            text="●",
            font=FONTS['body'],
            bg=COLORS['bg_light'],
            fg=COLORS['accent_green']
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        # Content
        content_frame = tk.Frame(item_frame, bg=COLORS['bg_light'])
        content_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(
            content_frame,
            text=title,
            font=FONTS['body'],
            bg=COLORS['bg_light'],
            fg=COLORS['text_primary']
        ).pack(anchor='w')
        
        tk.Label(
            content_frame,
            text=description,
            font=FONTS['small'],
            bg=COLORS['bg_light'],
            fg=COLORS['text_secondary']
        ).pack(anchor='w')
        
        # Time
        tk.Label(
            item_frame,
            text=time,
            font=FONTS['small'],
            bg=COLORS['bg_light'],
            fg=COLORS['text_secondary']
        ).pack(side=tk.RIGHT)


class ConfigurationView(BaseView):
    """Configuration view for bot settings."""
    
    def create_widgets(self):
        self.create_header("Configuration", "Configure your trading bot settings")
        
        # Scrollable content
        canvas = tk.Canvas(self, bg=COLORS['bg_dark'], highlightthickness=0)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=COLORS['bg_dark'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Exchange settings card
        exchange_card, exchange_content = self.create_card(
            scrollable_frame, "Exchange Settings"
        )
        exchange_card.pack(fill=tk.X, padx=30, pady=10)
        
        # Exchange inputs
        self.exchange_frame, self.exchange_var = self.create_combo(
            exchange_content, "Exchange", ["binance", "coinbase", "kraken", "kucoin"]
        )
        self.exchange_frame.pack(fill=tk.X, pady=5)
        
        self.api_key_frame, self.api_key_var = self.create_input(
            exchange_content, "API Key", show="*"
        )
        self.api_key_frame.pack(fill=tk.X, pady=5)
        
        self.api_secret_frame, self.api_secret_var = self.create_input(
            exchange_content, "API Secret", show="*"
        )
        self.api_secret_frame.pack(fill=tk.X, pady=5)
        
        # Trading settings card
        trading_card, trading_content = self.create_card(
            scrollable_frame, "Trading Settings"
        )
        trading_card.pack(fill=tk.X, padx=30, pady=10)
        
        self.symbol_frame, self.symbol_var = self.create_combo(
            trading_content, "Trading Pair", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
        )
        self.symbol_frame.pack(fill=tk.X, pady=5)
        
        self.timeframe_frame, self.timeframe_var = self.create_combo(
            trading_content, "Timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        )
        self.timeframe_frame.pack(fill=tk.X, pady=5)
        
        self.trade_amount_frame, self.trade_amount_var = self.create_input(
            trading_content, "Trade Size (%)", "2.0"
        )
        self.trade_amount_frame.pack(fill=tk.X, pady=5)
        
        self.stop_loss_frame, self.stop_loss_var = self.create_input(
            trading_content, "Stop Loss (%)", "2.0"
        )
        self.stop_loss_frame.pack(fill=tk.X, pady=5)
        
        self.take_profit_frame, self.take_profit_var = self.create_input(
            trading_content, "Take Profit (%)", "5.0"
        )
        self.take_profit_frame.pack(fill=tk.X, pady=5)
        
        # ML Settings card
        ml_card, ml_content = self.create_card(
            scrollable_frame, "AI/ML Settings"
        )
        ml_card.pack(fill=tk.X, padx=30, pady=10)
        
        self.train_interval_frame, self.train_interval_var = self.create_input(
            ml_content, "Training Interval (hours)", "12"
        )
        self.train_interval_frame.pack(fill=tk.X, pady=5)
        
        # GPU checkbox
        gpu_frame = tk.Frame(ml_content, bg=COLORS['bg_light'])
        gpu_frame.pack(fill=tk.X, pady=10)
        
        self.use_gpu_var = tk.BooleanVar(value=True)
        gpu_check = tk.Checkbutton(
            gpu_frame,
            text="Enable GPU Acceleration (RTX 3080)",
            variable=self.use_gpu_var,
            font=FONTS['body'],
            bg=COLORS['bg_light'],
            fg=COLORS['text_primary'],
            selectcolor=COLORS['bg_medium'],
            activebackground=COLORS['bg_light'],
            activeforeground=COLORS['accent_green']
        )
        gpu_check.pack(anchor='w')
        
        # Action buttons
        button_frame = tk.Frame(scrollable_frame, bg=COLORS['bg_dark'])
        button_frame.pack(fill=tk.X, padx=30, pady=20)
        
        self.create_button(button_frame, "▶ Start Trading Bot", self.start_bot).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        self.create_button(button_frame, "⏹ Stop Bot", self.stop_bot, 'danger').pack(
            side=tk.LEFT, padx=(0, 10)
        )
        self.create_button(button_frame, "🔌 Test Connection", self.test_connection, 'secondary').pack(
            side=tk.LEFT
        )
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_combo(self, parent, label: str, values: list):
        """Create a combobox with label."""
        frame = tk.Frame(parent, bg=COLORS['bg_light'])
        
        tk.Label(
            frame,
            text=label,
            font=FONTS['body'],
            bg=COLORS['bg_light'],
            fg=COLORS['text_secondary']
        ).pack(anchor='w', pady=(0, 5))
        
        # Style for combobox
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            'Matrix.TCombobox',
            fieldbackground=COLORS['bg_medium'],
            background=COLORS['bg_light'],
            foreground=COLORS['text_primary'],
            bordercolor=COLORS['border'],
            arrowcolor=COLORS['accent_green']
        )
        
        var = tk.StringVar(value=values[0] if values else "")
        combo = ttk.Combobox(
            frame,
            textvariable=var,
            values=values,
            font=FONTS['body'],
            style='Matrix.TCombobox',
            state='readonly'
        )
        combo.pack(fill=tk.X, ipady=6)
        
        return frame, var
    
    def start_bot(self):
        """Start the trading bot."""
        messagebox.showinfo(
            "Start Bot",
            "Trading bot will be started with configured settings.\n\nThis is a demo - full integration pending."
        )
    
    def stop_bot(self):
        """Stop the trading bot."""
        messagebox.showinfo(
            "Stop Bot",
            "Trading bot will be stopped."
        )
    
    def test_connection(self):
        """Test exchange connection."""
        messagebox.showinfo(
            "Test Connection",
            "Testing connection to exchange...\n\nThis is a demo - full integration pending."
        )


class MonitorView(BaseView):
    """Real-time trading monitor view."""
    
    def create_widgets(self):
        self.create_header("Trading Monitor", "Live market data and bot activity")
        
        # Create monitoring content
        container = tk.Frame(self, bg=COLORS['bg_dark'])
        container.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)
        
        # Market info card
        market_card, market_content = self.create_card(container, "Market Information")
        market_card.pack(fill=tk.X, pady=10)
        
        # Create info grid
        info_grid = tk.Frame(market_content, bg=COLORS['bg_light'])
        info_grid.pack(fill=tk.X)
        
        # Row 1
        self.create_info_item(info_grid, "Current Price:", "$--", 0, 0)
        self.create_info_item(info_grid, "Predicted Price:", "$--", 0, 2)
        
        # Row 2
        self.create_info_item(info_grid, "Market Sentiment:", "--", 1, 0)
        self.create_info_item(info_grid, "Trend:", "--", 1, 2)
        
        # Row 3
        self.create_info_item(info_grid, "Balance:", "$--", 2, 0)
        self.create_info_item(info_grid, "Total P&L:", "$--", 2, 2)
        
        # Recent trades card
        trades_card, trades_content = self.create_card(container, "Recent Trades")
        trades_card.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Trades table placeholder
        tk.Label(
            trades_content,
            text="No trades yet. Start the bot to begin trading.",
            font=FONTS['body'],
            bg=COLORS['bg_light'],
            fg=COLORS['text_secondary']
        ).pack(pady=50)
        
        # Refresh button
        refresh_btn = self.create_button(container, "🔄 Refresh Data", self.refresh_data, 'secondary')
        refresh_btn.pack(pady=10)
    
    def create_info_item(self, parent, label, value, row, col):
        """Create an info display item."""
        frame = tk.Frame(parent, bg=COLORS['bg_light'])
        frame.grid(row=row, column=col, padx=20, pady=10, sticky='w')
        
        tk.Label(
            frame,
            text=label,
            font=FONTS['small'],
            bg=COLORS['bg_light'],
            fg=COLORS['text_secondary']
        ).pack(anchor='w')
        
        tk.Label(
            frame,
            text=value,
            font=FONTS['heading'],
            bg=COLORS['bg_light'],
            fg=COLORS['text_matrix']
        ).pack(anchor='w')
        
    def refresh_data(self):
        """Refresh monitoring data."""
        messagebox.showinfo("Refresh", "Data refreshed")


class ChartsView(BaseView):
    """Charts and analysis view."""
    
    def create_widgets(self):
        self.create_header("Charts & Analysis", "Visual analysis and predictions")
        
        container = tk.Frame(self, bg=COLORS['bg_dark'])
        container.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)
        
        # Chart controls
        controls_frame = tk.Frame(container, bg=COLORS['bg_dark'])
        controls_frame.pack(fill=tk.X, pady=10)
        
        self.create_button(controls_frame, "📊 Generate Price Chart", self.generate_chart).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        self.create_button(controls_frame, "📈 Performance Dashboard", self.generate_dashboard, 'secondary').pack(
            side=tk.LEFT, padx=(0, 10)
        )
        self.create_button(controls_frame, "💾 Export History", self.export_history, 'secondary').pack(
            side=tk.LEFT
        )
        
        # Chart display area
        chart_card, chart_content = self.create_card(container, "Chart Visualization")
        chart_card.pack(fill=tk.BOTH, expand=True, pady=10)
        
        tk.Label(
            chart_content,
            text="📊\n\nGenerate a chart to display here",
            font=FONTS['subtitle'],
            bg=COLORS['bg_light'],
            fg=COLORS['text_secondary']
        ).pack(expand=True, pady=100)
    
    def generate_chart(self):
        """Generate price chart."""
        messagebox.showinfo("Generate Chart", "Generating price prediction chart...")
    
    def generate_dashboard(self):
        """Generate performance dashboard."""
        messagebox.showinfo("Generate Dashboard", "Generating performance dashboard...")
    
    def export_history(self):
        """Export trading history."""
        messagebox.showinfo("Export History", "Exporting trading history...")


class SystemView(BaseView):
    """System information view."""
    
    def create_widgets(self):
        self.create_header("System Information", "Hardware and software status")
        
        container = tk.Frame(self, bg=COLORS['bg_dark'])
        container.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)
        
        # Hardware info
        hw_card, hw_content = self.create_card(container, "Hardware Information")
        hw_card.pack(fill=tk.X, pady=10)
        
        hw_grid = tk.Frame(hw_content, bg=COLORS['bg_light'])
        hw_grid.pack(fill=tk.X)
        
        hw_info = [
            ("💻 CPU", "Intel i9-13900K"),
            ("🎮 GPU", "NVIDIA RTX 3080 (10GB VRAM)"),
            ("💾 RAM", "32 GB DDR5"),
            ("⚡ CUDA", "Enabled"),
        ]
        
        for i, (label, value) in enumerate(hw_info):
            self.create_info_row(hw_grid, label, value, i)
        
        # Software info
        sw_card, sw_content = self.create_card(container, "Software Versions")
        sw_card.pack(fill=tk.X, pady=10)
        
        sw_grid = tk.Frame(sw_content, bg=COLORS['bg_light'])
        sw_grid.pack(fill=tk.X)
        
        import sys
        import tensorflow as tf
        
        sw_info = [
            ("🐍 Python", f"{sys.version.split()[0]}"),
            ("🧠 TensorFlow", f"{tf.__version__}"),
            ("📦 NumPy", "Latest"),
            ("📊 Pandas", "Latest"),
        ]
        
        for i, (label, value) in enumerate(sw_info):
            self.create_info_row(sw_grid, label, value, i)
        
        # Resource monitoring
        resource_card, resource_content = self.create_card(container, "Resource Monitoring")
        resource_card.pack(fill=tk.X, pady=10)
        
        # Placeholder for resource meters
        tk.Label(
            resource_content,
            text="Real-time resource monitoring\nCPU • GPU • Memory • Temperature",
            font=FONTS['body'],
            bg=COLORS['bg_light'],
            fg=COLORS['text_secondary'],
            justify=tk.CENTER
        ).pack(pady=30)
        
        refresh_btn = self.create_button(resource_content, "🔄 Refresh", self.refresh, 'secondary')
        refresh_btn.pack()
    
    def create_info_row(self, parent, label, value, row):
        """Create an information row."""
        frame = tk.Frame(parent, bg=COLORS['bg_light'])
        frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            frame,
            text=label,
            font=FONTS['body'],
            bg=COLORS['bg_light'],
            fg=COLORS['text_secondary']
        ).pack(side=tk.LEFT, padx=(0, 20))
        
        tk.Label(
            frame,
            text=value,
            font=FONTS['body'],
            bg=COLORS['bg_light'],
            fg=COLORS['text_matrix']
        ).pack(side=tk.LEFT)
    
    def refresh(self):
        """Refresh system information."""
        messagebox.showinfo("Refresh", "System information refreshed")


class HistoryView(BaseView):
    """Trade history view."""
    
    def create_widgets(self):
        self.create_header("Trade History", "View and export your trading history")
        
        container = tk.Frame(self, bg=COLORS['bg_dark'])
        container.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)
        
        # History card
        history_card, history_content = self.create_card(container, "Trading History")
        history_card.pack(fill=tk.BOTH, expand=True, pady=10)
        
        tk.Label(
            history_content,
            text="📜\n\nNo trading history available yet.\nStart trading to see your history here.",
            font=FONTS['subtitle'],
            bg=COLORS['bg_light'],
            fg=COLORS['text_secondary'],
            justify=tk.CENTER
        ).pack(expand=True, pady=100)
        
        # Export buttons
        export_frame = tk.Frame(container, bg=COLORS['bg_dark'])
        export_frame.pack(fill=tk.X, pady=10)
        
        self.create_button(export_frame, "💾 Export as JSON", self.export_json, 'secondary').pack(
            side=tk.LEFT, padx=(0, 10)
        )
        self.create_button(export_frame, "📊 Export as CSV", self.export_csv, 'secondary').pack(
            side=tk.LEFT
        )
    
    def export_json(self):
        """Export history as JSON."""
        messagebox.showinfo("Export JSON", "Exporting trading history as JSON...")
    
    def export_csv(self):
        """Export history as CSV."""
        messagebox.showinfo("Export CSV", "Exporting trading history as CSV...")


def launch_matrix_gui():
    """Launch the Matrix-themed GUI application."""
    app = MatrixWindow()
    app.mainloop()


if __name__ == "__main__":
    launch_matrix_gui()
