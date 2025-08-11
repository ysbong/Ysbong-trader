import os, json, logging, asyncio, requests, sqlite3, time
import numpy as np
from flask import Flask
from threading import Thread
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ChatMember
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)
# Data Handling Imports
import pandas as pd
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import random
import math

# Type hinting imports
from typing import List, Tuple, Union, Optional, Dict

# === Channel Membership Requirement ===
CHANNEL_USERNAME = "@ProsperityEngines"  # Replace with your channel username
CHANNEL_LINK = "https://t.me/ProsperityEngines"  # Replace with your channel link

async def is_user_joined(user_id: int, bot) -> bool:
    """Checks if a user is a member of the required Telegram channel."""
    try:
        member = await bot.get_chat_member(chat_id=CHANNEL_USERNAME, user_id=user_id)
        return member.status in [ChatMember.MEMBER, ChatMember.OWNER, ChatMember.ADMINISTRATOR]
    except Exception as e:
        logging.error(f"Error checking membership for user {user_id}: {e}")
        return False

# For local development: Load environment variables from .env file
# On Render, environment variables are set directly in the dashboard.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass # dotenv is not required in production environment like Render

# === Flask ping for Render Uptime ===
web_app = Flask(__name__)

@web_app.route('/')
def home() -> str:
    """Root endpoint for the web application."""
    return "YSBONG TRADER™ (Active) is awake and scanning!"

@web_app.route("/health")
def health() -> Tuple[str, int]:
    """Health check endpoint."""
    return "✅ YSBONG™ is alive and kicking!", 200

def run_web() -> None:
    """Runs the Flask web application in a separate thread."""
    port = int(os.environ.get("PORT", 8080))
    web_app.run(host="0.0.0.0", port=port)

# Start the Flask app in a separate thread
Thread(target=run_web).start()

# === SQLite Learning Memory ===
DB_FILE = "ysbong_memory.db"
SQLITE_TIMEOUT = 10.0 # seconds

def init_db() -> None:
    """Initializes the SQLite database tables."""
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    pair TEXT,
                    timeframe TEXT,
                    action_for_db TEXT, -- 'BUY' or 'SELL' (or 'HOLD' if the logic returns it)
                    price REAL,
                    rsi REAL,
                    ema REAL,
                    ma REAL,
                    resistance REAL,
                    support REAL,
                    macd REAL,
                    macd_signal REAL,
                    stoch_k REAL,
                    stoch_d REAL,
                    atr REAL,
                    hma REAL,
                    t3 REAL,
                    feedback TEXT DEFAULT NULL, -- 'win' or 'loss'
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS user_api_keys (
                    user_id INTEGER PRIMARY KEY,
                    api_key TEXT NOT NULL
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS candle_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    candle_data TEXT NOT NULL,  -- JSON string of candle data
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
    except sqlite3.Error as e:
        logging.error(f"SQLite initialization error: {e}")

init_db()

# === Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

user_data: dict = {}
usage_count: dict = {}

# CANDLE MEMORY SYSTEM
CANDLE_MEMORY_SIZE = 100  # Keep last 100 candles in memory per pair+timeframe

def update_candle_memory(pair: str, timeframe: str, new_candles: List[dict]) -> None:
    """Updates candle memory with new candles, maintaining a rolling window."""
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            
            # Retrieve existing candle memory
            c.execute("SELECT candle_data FROM candle_memory WHERE pair = ? AND timeframe = ? ORDER BY timestamp DESC LIMIT 1", 
                      (pair, timeframe))
            row = c.fetchone()
            
            existing_candles = []
            if row:
                existing_candles = json.loads(row[0])
            
            # Add new candles to existing memory
            for candle in new_candles:
                # Skip if we already have this candle (based on datetime)
                if any(c['datetime'] == candle['datetime'] for c in existing_candles):
                    continue
                existing_candles.append(candle)
            
            # Trim to max memory size
            if len(existing_candles) > CANDLE_MEMORY_SIZE:
                existing_candles = existing_candles[-CANDLE_MEMORY_SIZE:]
            
            # Update database
            candle_json = json.dumps(existing_candles)
            # Use INSERT OR REPLACE to always have the latest state for the pair/timeframe
            c.execute("INSERT OR REPLACE INTO candle_memory (pair, timeframe, candle_data) VALUES (?, ?, ?)",
                      (pair, timeframe, candle_json))
            conn.commit()
            
    except sqlite3.Error as e:
        logger.error(f"Error updating candle memory: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON error in candle memory: {e}")

def get_candle_memory(pair: str, timeframe: str) -> List[dict]:
    """Retrieves candle memory for a specific pair and timeframe."""
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute("SELECT candle_data FROM candle_memory WHERE pair = ? AND timeframe = ? ORDER BY timestamp DESC LIMIT 1", 
                      (pair, timeframe))
            row = c.fetchone()
            if row:
                return json.loads(row[0])
    except sqlite3.Error as e:
        logger.error(f"Error retrieving candle memory: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in candle memory: {e}")
    return []

def load_saved_keys() -> dict:
    """Loads saved API keys from the database."""
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute("SELECT user_id, api_key FROM user_api_keys")
            keys = {str(row[0]): row[1] for row in c.fetchall()}
            return keys
    except sqlite3.Error as e:
        logger.error(f"Error loading API keys from DB: {e}")
        return {}

def save_keys(user_id: int, api_key: str) -> None:
    """Saves an API key to the database."""
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO user_api_keys (user_id, api_key) VALUES (?, ?)", (user_id, api_key))
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error saving API key to DB: {e}")

def remove_key(user_id: int) -> None:
    """Removes an API key from the database."""
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM user_api_keys WHERE user_id = ?", (user_id,))
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error removing API key from DB: {e}")

saved_keys: dict = load_saved_keys() # Initial load

# ===== FLAG MAPPING =====
# Corrected mapping to use currency codes as keys
CURRENCY_FLAGS = {
    "USD": "🇺🇸",
    "EUR": "🇪🇺",
    "GBP": "🇬🇧",
    "JPY": "🇯🇵",
    "CHF": "🇨🇭",
    "CAD": "🇨🇦",
    "AUD": "🇦🇺",
    "NZD": "🇳🇿",
}

def get_flagged_pair_name(pair: str) -> str:
    """Converts 'EUR/USD' to 'EUR/USD🇪🇺🇺🇸' with  spaces between flags or pair"""
    base, quote = pair.split("/")
    flag1 = CURRENCY_FLAGS.get(base, "")
    flag2 = CURRENCY_FLAGS.get(quote, "")
    return f" {pair}  /{flag1}/{flag2}             "  # Example: EUR/USD🇪🇺🇺🇸

# === Constants ===
PAIRS: List[str] = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "USD/CAD",
    "AUD/USD", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY",
    "EUR/AUD", "AUD/JPY", "CHF/JPY", "NZD/JPY", "EUR/CAD",
    "CAD/JPY", "GBP/CAD", "GBP/AUD", "AUD/CAD", "AUD/CHF"]
TIMEFRAMES: List[str] = ["1MIN", "5MIN", "15MIN"]

# === TwelveData API Fetcher ===

def fetch_data(api_key: str, symbol: str, interval: str = "1min", outputsize: int = 100) -> Tuple[str, Union[List[dict], str]]:
    """
    Fetches candlestick data from TwelveData.
    Returns a tuple: ("success", data) or ("error", error_message)
    """
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=10) # Added timeout
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if data.get("status") == "error":
            return "error", data.get("message", "Unknown API error.")
        
        candles = data.get("values", [])
        if not candles:
            return "error", "No data returned for the given symbol and interval. Market might be closed or invalid parameters."
        
        # TwelveData returns latest first, so reverse to have oldest first
        return "success", list(reversed(candles))

    except requests.exceptions.Timeout:
        return "error", "Request timed out. TwelveData API might be slow or unreachable."
    except requests.exceptions.HTTPError as e:
        return "error", f"HTTP error from TwelveData: {e}. Check API key or symbol."
    except requests.exceptions.ConnectionError:
        return "error", "Connection error. Could not reach TwelveData API."
    except requests.exceptions.RequestException as e:
        return "error", f"An unexpected request error occurred: {e}"
    except json.JSONDecodeError:
        return "error", "Failed to decode JSON response from TwelveData."
    except Exception as e:
        return "error", f"An unexpected error occurred during data fetch: {e}"

# === INDICATOR CALCULATIONS ===

def calculate_ema_series(data: List[float], period: int) -> List[float]:
    """Calculate Exponential Moving Average (EMA) series."""
    if not data:
        return []
    if len(data) < period:
        # If not enough data for the full period, return padded list with last value
        return [data[-1]] * len(data) if data else []
        
    k = 2 / (period + 1)
    ema_values = []
    
    # Initialize the first EMA value with the SMA for the first 'period' values
    ema = sum(data[:period]) / period
    ema_values.append(ema)

    for i in range(period, len(data)):
        ema = data[i] * k + ema * (1 - k)
        ema_values.append(ema)
    return ema_values

def calculate_ema(closes: List[float], period: int) -> float:
    """Calculate the latest Exponential Moving Average (EMA)."""
    series = calculate_ema_series(closes, period)
    return series[-1] if series else (closes[-1] if closes else 0.0)

def calculate_sma(data: List[float], window: int) -> float:
    """Calculate Simple Moving Average (SMA)."""
    if len(data) < window:
        return data[-1] if data else 0.0
    return sum(data[-window:]) / window

def calculate_rsi(closes: List[float], period: int = 14) -> float:
    """Calculate Relative Strength Index (RSI)."""
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]
    avg_gain = sum(gains[:period]) / period if gains else 0.0
    avg_loss = sum(losses[:period]) / period if losses else 0.0
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    for i in range(period, len(deltas)):
        gain = deltas[i] if deltas[i] > 0 else 0
        loss = -deltas[i] if deltas[i] < 0 else 0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 1000.0
        rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(closes: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[float, float]:
    """Calculate Moving Average Convergence Divergence (MACD)."""
    if len(closes) < slow_period + signal_period:
        return 0.0, 0.0
    
    ema_fast_series = calculate_ema_series(closes, fast_period)
    ema_slow_series = calculate_ema_series(closes, slow_period)

    # Ensure series are long enough for MACD calculation
    if not ema_fast_series or not ema_slow_series:
        return 0.0, 0.0

    # Align the series to their latest common point
    min_len = min(len(ema_fast_series), len(ema_slow_series))
    aligned_ema_fast = ema_fast_series[-min_len:]
    aligned_ema_slow = ema_slow_series[-min_len:]

    macd_line_series = [ef - es for ef, es in zip(aligned_ema_fast, aligned_ema_slow)]
    
    if not macd_line_series:
        return 0.0, 0.0
        
    macd_signal_series = calculate_ema_series(macd_line_series, signal_period)
    
    return macd_line_series[-1], macd_signal_series[-1] if macd_signal_series else 0.0

def calculate_stochastic(highs: List[float], lows: List[float], closes: List[float], k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
    """Calculate Stochastic Oscillator (%K and %D)."""
    if len(closes) < k_period + d_period:
        return 50.0, 50.0
    percent_k_values = []
    for i in range(k_period - 1, len(closes)):
        low = min(lows[i - k_period + 1: i + 1])
        high = max(highs[i - k_period + 1: i + 1])
        percent_k = 50.0 if high == low else ((closes[i] - low) / (high - low)) * 100
        percent_k_values.append(percent_k)
    percent_d_values = []
    for i in range(d_period - 1, len(percent_k_values)):
        percent_d_values.append(sum(percent_k_values[i - d_period + 1: i + 1]) / d_period)
    return percent_k_values[-1], percent_d_values[-1]

def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    """Calculate Average True Range (ATR)."""
    if len(closes) < period + 1:
        return 0.0
    tr_list = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        tr_list.append(tr)
    if len(tr_list) < period:
        return sum(tr_list) / len(tr_list) if tr_list else 0.0
    atr = sum(tr_list[:period]) / period
    for i in range(period, len(tr_list)):
        atr = ((atr * (period - 1)) + tr_list[i]) / period
    return atr

def calculate_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    """Calculate Average Directional Index (ADX)."""
    if len(closes) < period + 1:
        return 20.0
    tr_list, plus_dm_list, minus_dm_list = [], [], []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        tr_list.append(tr)
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)
    atr = calculate_atr(highs, lows, closes, period)
    
    # Ensure plus_dm_list and minus_dm_list are long enough for EMA calculation
    if not plus_dm_list or not minus_dm_list:
        return 20.0 # Default ADX if not enough data
        
    plus_di = 100 * (calculate_ema(plus_dm_list, period) / atr) if atr else 0.0
    minus_di = 100 * (calculate_ema(minus_dm_list, period) / atr) if atr else 0.0
    dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    return round(dx, 2)

# === NEW INDICATORS: HULL MA AND T3 MA ===

def calculate_wma_series(data: List[float], period: int) -> List[float]:
    """Calculate Weighted Moving Average (WMA) series."""
    if not data or len(data) < period:
        # If not enough data, return padded list with last value
        return [data[-1]] * len(data) if data else []
    
    wma_values = []
    weights = np.arange(1, period + 1)
    sum_weights = np.sum(weights)

    for i in range(period - 1, len(data)):
        segment = data[i - period + 1 : i + 1]
        wma = np.sum(weights * segment) / sum_weights
        wma_values.append(wma)
    return wma_values

def calculate_wma(data: List[float], window: int) -> float:
    """Calculate the latest Weighted Moving Average (WMA)."""
    series = calculate_wma_series(data, window)
    return series[-1] if series else (data[-1] if data else 0.0)

def calculate_hma(closes: List[float], period: int = 14) -> float:
    """Calculate Hull Moving Average (HMA)"""
    if len(closes) < period:
        return 0.0
    
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(math.sqrt(period)))
    
    # Calculate WMA series for half period and full period
    wma_half_series = calculate_wma_series(closes, half_period)
    wma_full_series = calculate_wma_series(closes, period)
    
    if not wma_half_series or not wma_full_series:
        return 0.0

    # Align the series to their latest common point
    # wma_full_series will be shorter than wma_half_series.
    # We take the latest part of wma_half_series that matches the length of wma_full_series.
    aligned_wma_half = wma_half_series[len(wma_half_series) - len(wma_full_series):]
    
    raw_hma_series = [2 * wh - wf for wh, wf in zip(aligned_wma_half, wma_full_series)]
    
    if not raw_hma_series:
        return 0.0
        
    # Calculate WMA of the raw HMA series
    hma_series = calculate_wma_series(raw_hma_series, sqrt_period)
    
    return hma_series[-1] if hma_series else 0.0

def calculate_t3(closes: List[float], period: int = 14, volume_factor: float = 0.7) -> float:
    """Calculate T3 Moving Average"""
    if len(closes) < period: # T3 needs a good amount of data for nested EMAs
        return 0.0
    
    # Calculate the six EMAs
    e1_series = calculate_ema_series(closes, period)
    if not e1_series: return 0.0
    
    e2_series = calculate_ema_series(e1_series, period)
    if not e2_series: return 0.0

    e3_series = calculate_ema_series(e2_series, period)
    if not e3_series: return 0.0

    e4_series = calculate_ema_series(e3_series, period)
    if not e4_series: return 0.0

    e5_series = calculate_ema_series(e4_series, period)
    if not e5_series: return 0.0

    e6_series = calculate_ema_series(e5_series, period)
    if not e6_series: return 0.0
    
    # Get the latest values of each EMA series
    e1 = e1_series[-1]
    e2 = e2_series[-1]
    e3 = e3_series[-1]
    e4 = e4_series[-1]
    e5 = e5_series[-1]
    e6 = e6_series[-1]
    
    # Calculate coefficients
    c1 = -volume_factor * volume_factor * volume_factor
    c2 = 3 * volume_factor * volume_factor + 3 * volume_factor * volume_factor * volume_factor
    c3 = -6 * volume_factor * volume_factor - 3 * volume_factor - 3 * volume_factor * volume_factor
    c4 = 1 + 3 * volume_factor + volume_factor * volume_factor * volume_factor + 3 * volume_factor * volume_factor
    
    # Calculate T3
    t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    return t3

# === IMPROVED TREND DETECTION ===

def detect_trend_bias_strong(
    closes: List[float], highs: List[float], lows: List[float],
    ema_period: int = 20, rsi_threshold: int = 55, adx_threshold: int = 20
) -> str:
    """Detects strong trend bias based on multiple indicators."""
    if len(closes) < ema_period + 5:
        return "neutral"

    # EMA direction
    # Ensure enough data for EMA calculation for both current and previous points
    if len(closes) < ema_period + 5:
        return "neutral"
        
    ema_now = calculate_ema(closes, ema_period)
    ema_prev = calculate_ema(closes[:-5], ema_period) # Compare with EMA 5 candles ago

    # RSI and ADX confirmation
    rsi = calculate_rsi(closes, 14)
    adx = calculate_adx(highs, lows, closes, 14)

    # MACD confirmation (optional)
    macd_line, macd_signal = calculate_macd(closes)
    macd_trend_up = macd_line > macd_signal
    macd_trend_down = macd_line < macd_signal

    # Check for uptrend
    if ema_now > ema_prev and rsi >= rsi_threshold and adx >= adx_threshold and macd_trend_up:
        return "uptrend"

    # Check for downtrend
    if ema_now < ema_prev and rsi <= (100 - rsi_threshold) and adx >= adx_threshold and macd_trend_down:
        return "downtrend"

    return "neutral"

def calculate_indicators(candles: List[dict]) -> Optional[dict]:
    """Calculates a set of technical indicators from candlestick data."""
    if not candles or len(candles) < 30: # Minimum candles needed for meaningful calculations
        closes = [float(c['close']) for c in candles]
        highs = [float(c['high']) for c in candles]
        lows = [float(c['low']) for c in candles]
        current = closes[-1] if closes else 0.0
        return {
            "MA": current, "EMA": current, "RSI": 50.0,
            "Resistance": max(highs) if highs else current,
            "Support": min(lows) if lows else current,
            "MACD": 0.0, "MACD_Signal": 0.0,
            "Stoch_K": 50.0, "Stoch_D": 50.0,
            "ATR": 0.0, "ADX": 20.0, "TrendBias": "neutral",
            "HMA": current,
            "T3": current
        }

    # Extract OHLC
    closes = [float(c['close']) for c in candles]
    highs = [float(c['high']) for c in candles]
    lows = [float(c['low']) for c in candles]

    # Calculate indicators
    ma = calculate_sma(closes, 20)
    ema = calculate_ema(closes, 20)
    rsi = calculate_rsi(closes)
    macd, macd_signal = calculate_macd(closes)
    stoch_k, stoch_d = calculate_stochastic(highs, lows, closes)
    atr = calculate_atr(highs, lows, closes)
    adx = calculate_adx(highs, lows, closes)
    
    # Calculate new indicators
    hma = calculate_hma(closes, 14)
    t3 = calculate_t3(closes, 14)

    # Use refined trend detector
    trend = detect_trend_bias_strong(closes, highs, lows)

    return {
        "MA": round(ma, 4),
        "EMA": round(ema, 4),
        "RSI": round(rsi, 2),
        "Resistance": round(max(highs), 4),
        "Support": round(min(lows), 4),
        "MACD": round(macd, 4),
        "MACD_Signal": round(macd_signal, 4),
        "Stoch_K": round(stoch_k, 2),
        "Stoch_D": round(stoch_d, 2),
        "ATR": round(atr, 4),
        "ADX": round(adx, 2),
        "TrendBias": trend,
        "HMA": round(hma, 4),
        "T3": round(t3, 4)
    }

def validate_signal_based_on_trend(indicators: dict, closes: List[float]) -> str:
    """Refined function to validate a potential signal based on trend and indicator values."""
    trend_bias = indicators.get("TrendBias", "neutral")
    adx = indicators.get("ADX", 20.0)
    rsi = indicators.get("RSI", 50.0)
    stoch_k = indicators.get("Stoch_K", 50.0)
    macd = indicators.get("MACD", 0.0)
    macd_signal = indicators.get("MACD_Signal", 0.0)

    # Simple check to see if MACD is bullish or bearish
    macd_is_bullish = macd > macd_signal
    macd_is_bearish = macd < macd_signal

    # --- Trend Filtering: This is the most important part ---
    # In a strong downtrend, only consider 'sell' signals.
    if trend_bias == "downtrend" and adx > 25:
        # Check for confirmation of the downtrend from indicators
        if rsi < 50 and stoch_k < 50 and macd_is_bearish:
            # Look for a re-entry point or continuation. A sell signal is valid if indicators are not oversold yet.
            if rsi > 30 and stoch_k > 20: # To avoid selling at the bottom
                return "sell"
        return "hold" # In a strong downtrend, we never 'buy'

    # In a strong uptrend, only consider 'buy' signals.
    elif trend_bias == "uptrend" and adx > 25:
        # Check for confirmation of the uptrend from indicators
        if rsi > 50 and stoch_k > 50 and macd_is_bullish:
            # Look for a re-entry point or continuation. A buy signal is valid if indicators are not overbought yet.
            if rsi < 70 and stoch_k < 80: # To avoid buying at the top
                return "buy"
        return "hold" # In a strong uptrend, we never 'sell'

    # --- Neutral or Weak Trend Logic ---
    # This section handles sideways or low-momentum markets.
    # We will look for classic mean-reversion signals (buying low, selling high).
    if adx <= 25:
        # Check for oversold conditions for a potential 'buy' signal
        if rsi < 30 and stoch_k < 20 and macd_is_bullish:
            return "buy"
        # Check for overbought conditions for a potential 'sell' signal
        elif rsi > 70 and stoch_k > 80 and macd_is_bearish:
            return "sell"
        
    return "hold" # If no clear signal is found, do nothing.

# === Telegram Handlers ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /start command."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    # Check if user has joined the channel
    if not await is_user_joined(user_id, context.bot):
        keyboard = [
            [InlineKeyboardButton("📢 Join Channel", url=CHANNEL_LINK)],
            [InlineKeyboardButton("✅ I Joined", callback_data="check_joined")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "🚫 You must join our channel to use this bot.\n\nPlease click the button below to join:",
            reply_markup=reply_markup
        )
        return

    # If user has joined, proceed with normal start flow
    user_data[user_id] = {}
    usage_count[user_id] = usage_count.get(user_id, 0)
    
    api_key_from_db = load_saved_keys().get(str(user_id))

    if api_key_from_db:
        user_data[user_id]["api_key"] = api_key_from_db
        kb = []
        for i in range(0, len(PAIRS), 2): 
                    row_buttons = [InlineKeyboardButton(get_flagged_pair_name(PAIRS[j]), callback_data=f"pair|{PAIRS[j]}") 
                                for j in range(i, min(i+2, len(PAIRS)))]
                    kb.append(row_buttons)

        await update.message.reply_text("🔑 API key loaded.\n💱 Choose Pair:", reply_markup=InlineKeyboardMarkup(kb))
        return

    kb = [[InlineKeyboardButton("✅ I Understand", callback_data="agree_disclaimer")]]
    await update.message.reply_text(
        "⚠️ DISCLAIMER\nThis bot provides educational signals only.\nYou are the engine of your prosperity.",
        reply_markup=InlineKeyboardMarkup(kb)
    )

async def check_joined_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles callback for checking channel membership."""
    query = update.callback_query
    user_id = query.from_user.id
    chat_id = query.message.chat_id
    await query.answer()

    if query.data == "check_joined":
        if await is_user_joined(user_id, context.bot):
            try:
                await query.message.delete()
            except Exception as e:
                logger.warning(f"Could not delete message for user {user_id} in check_joined_callback: {e}")
            
            # Proceed with normal start flow
            user_data[user_id] = {}
            usage_count[user_id] = usage_count.get(user_id, 0)
            
            api_key_from_db = load_saved_keys().get(str(user_id))

            if api_key_from_db:
                user_data[user_id]["api_key"] = api_key_from_db
                kb = []
                for i in range(0, len(PAIRS), 2): 
                    row_buttons = [InlineKeyboardButton(get_flagged_pair_name(PAIRS[j]), callback_data=f"pair|{PAIRS[j]}") 
                                for j in range(i, min(i+2, len(PAIRS)))]
                    kb.append(row_buttons)

                await context.bot.send_message(chat_id, "🔑 API key loaded.\n💱 Choose Pair:", reply_markup=InlineKeyboardMarkup(kb))
                return

            kb = [[InlineKeyboardButton("✅ I Understand", callback_data="agree_disclaimer")]]
            await context.bot.send_message(
                chat_id,
                "⚠️ DISCLAIMER\nThis bot provides educational signals only.\nYou are the engine of your prosperity.",
                reply_markup=InlineKeyboardMarkup(kb)
            )
        else:
            await query.answer("❗ You still haven't joined the channel. Please join and then click the button again.", show_alert=True)

async def howto(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provides instructions on how to use the bot."""
    reminder = await get_friendly_reminder()
    await update.message.reply_text(reminder, parse_mode='Markdown')

async def get_friendly_reminder() -> str:
    """Returns the formatted how-to message."""
    return (
        "📌 *Welcome to YSBONG TRADER™ – Friendly Reminder* 💬\n\n"
        "Hello Trader 👋\n\n"
        "Here’s how to get started with your *real live signals* (not simulation or OTC):\n\n"
        "🧑‍🏫 *How to Use the Bot*\n"
        "1. 🔑 Get your API key from https://twelvedata.com\n"
        "   → Register, log in, dashboard > API Key\n"
        "2. Copy your API KEY || Return to the bot\n"
        "3. Tap the menu button || Tap start\n"
        "4. ✅ Agree to the Disclaimer\n"   
        "   → Paste it here in the bot\n"
        "5. 💱 Choose Trading Pair & Timeframe\n"
        "6. ⚡ Click 📶 GET SIGNAL\n\n"
        "📢 *Note:*\n"
        "🏦 This is not OTC. Signals are based on real market data using your API key.\n"
        "🧠 Results depend on live charts.\n\n"
        "⚠️ *No trading on weekends* - the market is closed for non-OTC assets.\n"
        "🙇 *Beginners:*\n"
        "🧑‍💻  Practice first — observe signals.\n"
        "👉 Register here: https://pocket-friends.com/r/w2enb3tukw\n"
        "💵 Deposit when you're confident (min $10).\n\n"
        
        " 🔑 *About TwelveData API Key*\n" 

        "YSBONG TRADER™ uses real-time market data powered by [TwelveData](https://twelvedata.com).\n"
        "You’ll need an API key to activate signals.\n"
        "🆓 **Free Tier (Default when you register)** \n"
        "- ⏱️ Up to 800 API calls per day\n"
        "- 🔄 Max 8 requests per minute\n\n"

        "✌️✌️ GOOD LUCK TRADER ✌️✌️\n\n"

        "🤗 *Be patient. Be disciplined.*\n"
        "😋 *Greedy traders don't last-the market eats them alive.*\n"
        "Respect the market.\n"
        "– *YSBONG TRADER™ powered by PROSPERITY ENGINES™* 💪"
    )

async def disclaimer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Displays the financial risk disclaimer."""
    disclaimer_msg = (
    "⚠️ *Financial Risk Disclaimer*\n\n"
    "Trading involves real risk. This bot provides educational signals only.\n"
    "*Not financial advice.*\n\n"
    "🤔 Be wise. Only trade what you can afford to lose.\n"
    "🎯 Results depend on your discipline, not predictions.\n\n"
    "☣️ *Avoid overtrading!* More trades don’t mean more profits — they usually mean more mistakes.\n"
    "⏳🤚 Wait for clean setups, and trust the process.\n"
)
    await update.message.reply_text(disclaimer_msg, parse_mode='Markdown')

async def handle_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles inline keyboard button presses."""
    query = update.callback_query
    user_id = query.from_user.id
    chat_id = query.message.chat_id
    await query.answer()
    
    try:
        await query.message.delete()
    except Exception as e:
        logger.warning(f"Could not delete message for user {user_id} in handle_buttons: {e}")
        pass

    data = query.data
    if data == "agree_disclaimer":
        await context.bot.send_message(chat_id, "🔐 Please enter your API key:")
        user_data[user_id] = {"step": "awaiting_api"}
    elif data.startswith("pair|"):
        user_data[user_id]["pair"] = data.split("|")[1]
        kb = [[InlineKeyboardButton(tf, callback_data=f"timeframe|{tf}")] for tf in TIMEFRAMES]
        await context.bot.send_message(chat_id, "⏰ Choose Timeframe:", reply_markup=InlineKeyboardMarkup(kb))
    elif data.startswith("timeframe|"):
        user_data[user_id]["timeframe"] = data.split("|")[1]
        await context.bot.send_message(
            chat_id,
            "✅ Ready to generate signal!",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📶 GET SIGNAL 📶", callback_data="get_signal")]])
        )
    elif data == "get_signal":
        await generate_signal(update, context)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming text messages, primarily for API key input."""
    user_id = update.effective_user.id
    text = update.message.text.strip()
    chat_id = update.effective_chat.id

    if user_data.get(user_id, {}).get("step") == "awaiting_api":
        user_data[user_id]["api_key"] = text
        user_data[user_id]["step"] = None
        save_keys(user_id, text)
        kb = []
        for i in range(0, len(PAIRS), 2): 
            row_buttons = [InlineKeyboardButton(get_flagged_pair_name(PAIRS[j]), callback_data=f"pair|{PAIRS[j]}") 
                           for j in range(i, min(i + 2, len(PAIRS)))]
            kb.append(row_buttons)
        await context.bot.send_message(chat_id, "🔐 API Key saved.\n💱 Choose Currency Pair:", reply_markup=InlineKeyboardMarkup(kb))

async def generate_signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generates and sends a trading signal to the user."""
    query = update.callback_query
    user_id = query.from_user.id
    chat_id = query.message.chat_id
    usage_count[user_id] = usage_count.get(user_id, 0) + 1
    
    data = user_data.get(user_id, {})
    pair = data.get("pair")
    tf = data.get("timeframe")
    api_key = data.get("api_key")

    if not all([pair, tf, api_key]):
        await context.bot.send_message(chat_id, text="❌ Please set your API Key, Pair, and Timeframe first using /start.")
        return

    loading_msg = await context.bot.send_message(chat_id=chat_id, text="🧐 Analyzing market data...")
    
    # First try to get from memory
    memory_candles = get_candle_memory(pair, tf)
    
    # Fetch new data from API
    status, new_candles = fetch_data(api_key, pair)
    if status == "error":
        # If we have memory candles, use them with a warning
        if memory_candles:
            await loading_msg.edit_text(f"⚠️ Using cached data: {new_candles}")
            candles_to_use = memory_candles
        else:
            await loading_msg.edit_text(f"❌ Error fetching data: {new_candles}")
            if "API Key" in new_candles or "rate limit" in new_candles.lower():
                user_data[user_id].pop("api_key", None)
                remove_key(user_id)
                user_data[user_id]["step"] = "awaiting_api"
            return
    else:
        candles_to_use = new_candles
        # Update memory with new candles
        update_candle_memory(pair, tf, new_candles)
    
    # If we have both memory and new candles, combine them (memory update handles deduplication)
    if not candles_to_use:
        await loading_msg.edit_text(f"⚠️ No market data available for {pair} on {tf}. The market might be closed or data is unavailable.")
        return

    indicators = calculate_indicators(candles_to_use)
    
    if not indicators:
        await loading_msg.edit_text(f"❌ Could not calculate indicators for {pair}. Insufficient or malformed data.")
        return

    current_price = float(candles_to_use[-1]["close"])

    # Generate signal using enhanced rule-based system
    action_rule = validate_signal_based_on_trend(indicators, [float(c['close']) for c in candles_to_use])
    
    if action_rule == "buy":
        action = "BUY BUY BUY 👆👆👆"
        action_for_db = "BUY"
    elif action_rule == "sell":
        action = "SELL SELL SELL 👇👇👇"
        action_for_db = "SELL"
    else:  # hold
        # Use simple RSI-based fallback
        if indicators['RSI'] > 50:
            action = "BUY BUY BUY 👆👆👆"
            action_for_db = "BUY"
        else:
            action = "SELL SELL SELL 👇👇👇"
            action_for_db = "SELL"

    await loading_msg.delete()
    
    flagged_pair = get_flagged_pair_name(pair)
    
    signal = (
        f"🥸 *YSBONG TRADER™ SIGNAL* 🥸\n\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"💰 *PAIR:* `{flagged_pair}`\n"
        f"⏱️ *TIMEFRAME:* `{tf}`\n"
        f"🕺 *ACTION:* **{action}**\n"
        f"━━━━━━━━━━━━━━━━━━━\n\n"
        f"📊 *Current Market Data:*\n"
        f"🪙  Price: `{current_price}`\n\n"
        f"🔑 *Key Indicators:*\n"
        f"   • MA: `{indicators['MA']}`\n"
        f"   • EMA: `{indicators['EMA']}`\n"
        f"   • RSI: `{indicators['RSI']}`\n"
        f"   • Resistance: `{indicators['Resistance']}`\n"
        f"   • Support: `{indicators['Support']}`\n\n"
        f"🚀💥 *Advanced Indicators:*\n"
        f"   • MACD: `{indicators['MACD']}` (Signal: `{indicators['MACD_Signal']}`)\n"
        f"   • Stoch %K: `{indicators['Stoch_K']}` (Stoch %D: `{indicators['Stoch_D']}`)\n"
        f"   • ATR: `{indicators['ATR']}` (Volatility)\n"
        f"   • HMA: `{indicators['HMA']}`\n"
        f"   • T3: `{indicators['T3']}`\n"
        f"━━━━━━━━━━━━━━━━━━━\n\n"
        f"💡🫵 *Remember:* Always exercise caution and manage your risk. This is for educational purposes."
    )
    
    feedback_keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🤑 Win", callback_data=f"feedback|win"),
         InlineKeyboardButton("😭 Loss", callback_data=f"feedback|loss")]
    ])
    
    await context.bot.send_message(chat_id=chat_id, text=signal, parse_mode='Markdown', reply_markup=feedback_keyboard)
    
    # Store the signal, which will now always be BUY or SELL
    if action_for_db:
        store_signal(user_id, pair, tf, action_for_db, current_price,
                     indicators["RSI"], indicators["EMA"], indicators["MA"],
                     indicators["Resistance"], indicators["Support"],
                     indicators["MACD"], indicators["MACD_Signal"],
                     indicators["Stoch_K"], indicators["Stoch_D"], indicators["ATR"],
                     indicators["HMA"], indicators["T3"])

def store_signal(user_id: int, pair: str, tf: str, action: str, price: float,
                 rsi: float, ema: float, ma: float, resistance: float, support: float,
                 macd: float, macd_signal: float, stoch_k: float, stoch_d: float, atr: float,
                 hma: float, t3: float) -> None:
    """Stores a generated signal into the database."""
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO signals (user_id, pair, timeframe, action_for_db, price, rsi, ema, ma, resistance, support, macd, macd_signal, stoch_k, stoch_d, atr, hma, t3)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, pair, tf, action, price, rsi, ema, ma, resistance, support, macd, macd_signal, stoch_k, stoch_d, atr, hma, t3))
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error storing signal to DB: {e}")

async def reset_api(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Resets the user's stored API key."""
    user_id = update.effective_user.id
    
    api_key_exists = load_saved_keys().get(str(user_id))

    if api_key_exists:
        remove_key(user_id)
        if user_id in user_data:
            user_data[user_id].pop("api_key", None)
            user_data[user_id]["step"] = "awaiting_api"
        await update.message.reply_text("🗑️ API key removed. Please enter your new API key now or use /start to set a new one.")
    else:
        await update.message.reply_text("ℹ️ No API key found to reset.")

async def feedback_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles feedback (win/loss) provided by the user."""
    query = update.callback_query
    user_id = query.from_user.id
    chat_id = query.message.chat_id
    await query.answer()
    
    data = query.data.split('|')
    if data[0] == "feedback":
        feedback_result = data[1]
        
        try:
            with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
                c = conn.cursor()
                c.execute("SELECT id FROM signals WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1", (user_id,))
                row = c.fetchone()
                if row:
                    signal_id = row[0]
                    c.execute("UPDATE signals SET feedback = ? WHERE id = ?", (feedback_result, signal_id))
                    conn.commit()
                    logger.info(f"Feedback saved for signal {signal_id}: {feedback_result}")
                else:
                    logger.warning(f"No previous signal found for user {user_id} to apply feedback.")
        except sqlite3.Error as e:
            logger.error(f"Error saving feedback: {e}")

        try:
            await query.edit_message_text(f"✅ Feedback saved: **{feedback_result.upper()}**. Thank you for your input. I LOVE YOU😘😘😘!", parse_mode='Markdown')
        except Exception as e:
            logger.warning(f"Could not edit message for feedback for user {user_id}: {e}")
            await context.bot.send_message(chat_id, f"✅ Feedback saved: **{feedback_result.upper()}**. Thank you for your input. I LOVE YOU😘😘😘!")

# === New Features ===
INTRO_MESSAGE = """
📢 WELCOME TO YSBONG TRADER™ – SIGNAL SCANNER 📡

✍️ Designed to guide both beginners and experienced traders through real-time market signals.

🫣 What to Expect:
🔄 Auto-generated signals (BUY/SELL)
🕯️ Smart detection from indicators + candle logic
⚡ Fast, clean, no-hype trading alerts

💾 Feedback? Use the Win/Loss buttons  
→ Your result helps improve future signals

👭 Invite your friends to join:
https://t.me/ProsperityEngines

🤓 Trade smart. Stay focused. Respect the charts.
🐲 Let the PROSPERITY ENGINE help you sharpen your instincts.

— YSBONG TRADER™  
“SIGNAL SENT. PROSPERITY LOADED.”
"""

async def intro_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends the bot's introduction message."""
    await update.message.reply_text(INTRO_MESSAGE)

def get_all_users() -> List[int]:
    """Retrieves all unique user IDs from the user_api_keys table."""
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute("SELECT DISTINCT user_id FROM user_api_keys")
            users = [row[0] for row in c.fetchall()]
        return users
    except sqlite3.Error as e:
        logger.error(f"Error fetching all users: {e}")
        return []

async def send_intro_to_all_users(app: ApplicationBuilder) -> None:
    """Sends the introduction message to all users with saved API keys."""
    users = get_all_users()
    if not users:
        logger.info("No users found to send intro message")
        return
        
    logger.info(f"Attempting to send intro message to {len(users)} users...")
    
    for user_id in users:
        try:
            # Check if bot can send messages to this user (e.g., they haven't blocked the bot)
            chat_member = await app.bot.get_chat_member(chat_id=user_id, user_id=app.bot.id)
            if chat_member.status in [ChatMember.KICKED, ChatMember.LEFT]:
                logger.info(f"Skipping intro message for user {user_id}: Bot is blocked or user left chat.")
                continue

            await app.bot.send_message(chat_id=user_id, text=INTRO_MESSAGE)
            logger.info(f"✅ Intro sent to user: {user_id}")
            await asyncio.sleep(0.1) # Small delay to avoid hitting Telegram API limits
        except Exception as e:
            logger.warning(f"❌ Failed to send intro to {user_id}: {e}")

# === Start Bot ===
if __name__ == '__main__':
    # IMPORTANT: Load token from environment variable
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

    if not TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set. Bot cannot start.")
        print("ERROR: TELEGRAM_BOT_TOKEN environment variable not set. Please set it or add it to a .env file.")
        exit(1)

    app = ApplicationBuilder().token(TOKEN).build()

    # Add command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("howto", howto))
    app.add_handler(CommandHandler("disclaimer", disclaimer))
    app.add_handler(CommandHandler("resetapikey", reset_api))
    app.add_handler(CommandHandler("intro", intro_command))

    # Add other handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_buttons, pattern="^(pair|timeframe|get_signal|agree_disclaimer).*"))
    app.add_handler(CallbackQueryHandler(feedback_callback_handler, pattern=r"^feedback\|(win|loss)$"))
    app.add_handler(CallbackQueryHandler(check_joined_callback, pattern="^check_joined$"))

    # Setup scheduled intro message
    scheduler = BackgroundScheduler()
    # Schedule to run every Monday at 9 AM local time (adjust as needed for server timezone)
    scheduler.add_job(lambda: asyncio.run(send_intro_to_all_users(app)), 'cron', day_of_week='mon', hour=9)
    scheduler.start()
    logger.info("⏰ Scheduled weekly intro message configured (Mondays at 9 AM)")

    logger.info("✅ YSBONG TRADER™ is LIVE...")
    app.run_polling(drop_pending_updates=True)
