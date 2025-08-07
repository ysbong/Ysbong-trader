import os, json, logging, asyncio, requests, sqlite3, joblib, time
import numpy as np
from flask import Flask
from threading import Thread
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ChatMember
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)
# AI & Data Handling Imports
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from deap import base, creator, tools, algorithms
import random
import math
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline

# Type hinting imports
from typing import List, Tuple, Union, Optional, Dict

# === Channel Membership Requirement ===
CHANNEL_USERNAME = "@ProsperityEngines"
CHANNEL_LINK = "https://t.me/ProsperityEngines"

async def is_user_joined(user_id: int, bot) -> bool:
    try:
        member = await bot.get_chat_member(chat_id=CHANNEL_USERNAME, user_id=user_id)
        return member.status in [ChatMember.MEMBER, ChatMember.OWNER, ChatMember.ADMINISTRATOR]
    except Exception as e:
        logging.error(f"Error checking membership for user {user_id}: {e}")
        return False

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# === Flask ping for Render Uptime ===
web_app = Flask(__name__)

@web_app.route('/')
def home() -> str:
    return "🤖 YSBONG TRADER™ (AI Brain Active) is awake and learning!"

@web_app.route("/health")
def health() -> Tuple[str, int]:
    return "✅ YSBONG™ is alive and kicking!", 200

def run_web() -> None:
    port = int(os.environ.get("PORT", 8080))
    web_app.run(host="0.0.0.0", port=port)

# Start the Flask app in a separate thread
Thread(target=run_web).start()

# === SQLite Learning Memory ===
DB_FILE = "ysbong_memory.db"
MODEL_FILE = "ai_brain_model.joblib"
SQLITE_TIMEOUT = 10.0

def init_db() -> None:
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    pair TEXT,
                    timeframe TEXT,
                    action_for_db TEXT,
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
                    adx REAL,
                    feedback TEXT DEFAULT NULL,
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
                    candle_data TEXT NOT NULL,
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

# ========================
# === CANDLE MEMORY SYSTEM
# ========================
CANDLE_MEMORY_SIZE = 100

def update_candle_memory(pair: str, timeframe: str, new_candles: List[dict]) -> None:
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute("SELECT candle_data FROM candle_memory WHERE pair = ? AND timeframe = ? ORDER BY timestamp DESC LIMIT 1", 
                      (pair, timeframe))
            row = c.fetchone()
            
            existing_candles = []
            if row:
                existing_candles = json.loads(row[0])
            
            for candle in new_candles:
                if any(c['datetime'] == candle['datetime'] for c in existing_candles):
                    continue
                existing_candles.append(candle)
            
            if len(existing_candles) > CANDLE_MEMORY_SIZE:
                existing_candles = existing_candles[-CANDLE_MEMORY_SIZE:]
            
            candle_json = json.dumps(existing_candles)
            c.execute("INSERT INTO candle_memory (pair, timeframe, candle_data) VALUES (?, ?, ?)",
                      (pair, timeframe, candle_json))
            conn.commit()
            
    except sqlite3.Error as e:
        logger.error(f"Error updating candle memory: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON error in candle memory: {e}")

def get_candle_memory(pair: str, timeframe: str) -> List[dict]:
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
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO user_api_keys (user_id, api_key) VALUES (?, ?)", (user_id, api_key))
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error saving API key to DB: {e}")

def remove_key(user_id: int) -> None:
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM user_api_keys WHERE user_id = ?", (user_id,))
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error removing API key from DB: {e}")

saved_keys: dict = load_saved_keys()

# ===== FLAG MAPPING =====
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
    base, quote = pair.split("/")
    flag1 = CURRENCY_FLAGS.get(base, "")
    flag2 = CURRENCY_FLAGS.get(quote, "")
    return f"{pair}{flag1}{flag2}"

# === Constants ===
PAIRS: List[str] = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "USD/CAD",
    "AUD/USD", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY",
    "EUR/AUD", "AUD/JPY", "CHF/JPY", "NZD/JPY", "EUR/CAD",
    "CAD/JPY", "GBP/AUD", "AUD/CAD"]
TIMEFRAMES: List[str] = ["1MIN", "5MIN", "15MIN"]
MIN_FEEDBACK_FOR_TRAINING: int = 15
FEEDBACK_BATCH_SIZE: int = 5

# === TwelveData API Fetcher ===

def fetch_data(api_key: str, symbol: str, interval: str = "1min", outputsize: int = 100) -> Tuple[str, Union[List[dict], str]]:
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "error":
            return "error", data.get("message", "Unknown API error.")
        
        candles = data.get("values", [])
        if not candles:
            return "error", "No data returned for the given symbol and interval. Market might be closed or invalid parameters."
        
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

def calculate_ema(closes: List[float], period: int) -> float:
    if len(closes) < period:
        return closes[-1] if closes else 0.0
    k = 2 / (period + 1)
    ema = closes[0]
    for price in closes[1:]:
        ema = price * k + ema * (1 - k)
    return ema

def calculate_sma(data: List[float], window: int) -> float:
    if len(data) < window:
        return data[-1] if data else 0.0
    return sum(data[-window:]) / window

def calculate_rsi(closes: List[float], period: int = 14) -> float:
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
    if len(closes) < slow_period + signal_period:
        return 0.0, 0.0
    ema_fast = closes[0]
    ema_slow = closes[0]
    k_fast = 2 / (fast_period + 1)
    k_slow = 2 / (slow_period + 1)
    macd_line = []
    for price in closes:
        ema_fast = price * k_fast + ema_fast * (1 - k_fast)
        ema_slow = price * k_slow + ema_slow * (1 - k_slow)
        macd_line.append(ema_fast - ema_slow)
    signal_ema = macd_line[0]
    k_signal = 2 / (signal_period + 1)
    macd_signal = []
    for val in macd_line:
        signal_ema = val * k_signal + signal_ema * (1 - k_signal)
        macd_signal.append(signal_ema)
    return macd_line[-1], macd_signal[-1]

def calculate_stochastic(highs: List[float], lows: List[float], closes: List[float], k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
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
    plus_di = 100 * (calculate_ema(plus_dm_list, period) / atr) if atr else 0.0
    minus_di = 100 * (calculate_ema(minus_dm_list, period) / atr) if atr else 0.0
    dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    return round(dx, 2)

def calculate_wma(data: List[float], period: int) -> float:
    if len(data) < period:
        return 0.0
    weights = np.arange(1, period + 1)
    wma = np.sum(weights * data[-period:]) / np.sum(weights)
    return wma

def calculate_hma(closes: List[float], period: int = 14) -> float:
    if len(closes) < period:
        return 0.0
    
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(math.sqrt(period)))
    
    wma_half = calculate_wma(closes, half_period)
    wma_full = calculate_wma(closes, period)
    
    raw_hma = 2 * wma_half - wma_full
    
    adjusted_data = closes[-sqrt_period:]
    if len(adjusted_data) > 0:
        adjusted_data[-1] = raw_hma
    else:
        adjusted_data = [raw_hma]
    
    return calculate_wma(adjusted_data, sqrt_period)

def calculate_t3(closes: List[float], period: int = 14, volume_factor: float = 0.7) -> float:
    if len(closes) < period:
        return 0.0
    
    e1 = calculate_ema(closes, period)
    e2 = calculate_ema([e1] * len(closes), period)
    e3 = calculate_ema([e2] * len(closes), period)
    e4 = calculate_ema([e3] * len(closes), period)
    e5 = calculate_ema([e4] * len(closes), period)
    e6 = calculate_ema([e5] * len(closes), period)
    
    c1 = -volume_factor * volume_factor * volume_factor
    c2 = 3 * volume_factor * volume_factor + 3 * volume_factor * volume_factor * volume_factor
    c3 = -6 * volume_factor * volume_factor - 3 * volume_factor - 3 * volume_factor * volume_factor * volume_factor
    c4 = 1 + 3 * volume_factor + volume_factor * volume_factor * volume_factor + 3 * volume_factor * volume_factor
    
    t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    return t3

# === IMPROVED TREND DETECTION ===

def detect_trend_bias_strong(
    closes: List[float], highs: List[float], lows: List[float],
    ema_period: int = 20, rsi_threshold: int = 55, adx_threshold: int = 20
) -> str:
    if len(closes) < ema_period + 5:
        return "neutral"

    ema_now = calculate_ema(closes[-ema_period:], ema_period)
    ema_prev = calculate_ema(closes[-ema_period - 5:-5], ema_period)

    rsi = calculate_rsi(closes, 14)
    adx = calculate_adx(highs, lows, closes, 14)

    macd_line, macd_signal = calculate_macd(closes)
    macd_trend_up = macd_line > macd_signal

    if ema_now > ema_prev and rsi > rsi_threshold and adx >= adx_threshold and macd_trend_up:
        return "uptrend"

    if ema_now < ema_prev and rsi < (100 - rsi_threshold) and adx >= adx_threshold and not macd_trend_up:
        return "downtrend"

    return "neutral"

def calculate_indicators(candles: List[dict]) -> Optional[dict]:
    if not candles or len(candles) < 30:
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

    closes = [float(c['close']) for c in candles]
    highs = [float(c['high']) for c in candles]
    lows = [float(c['low']) for c in candles]

    ma = calculate_sma(closes, 20)
    ema = calculate_ema(closes, 20)
    rsi = calculate_rsi(closes)
    macd, macd_signal = calculate_macd(closes)
    stoch_k, stoch_d = calculate_stochastic(highs, lows, closes)
    atr = calculate_atr(highs, lows, closes)
    adx = calculate_adx(highs, lows, closes)
    hma = calculate_hma(closes, 14)
    t3 = calculate_t3(closes, 14)

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

# === GENETIC OPTIMIZER ===
def setup_genetic_algorithm():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    
    toolbox.register("attr_n_estimators", random.randint, 100, 500)
    toolbox.register("attr_max_depth", random.randint, 5, 30)
    toolbox.register("attr_learning_rate", random.uniform, 0.001, 0.1)
    toolbox.register("attr_hidden_layers", random.randint, 50, 300)
    toolbox.register("attr_alpha", random.uniform, 0.0001, 0.1)
    toolbox.register("attr_max_features", random.choice, ['auto', 'sqrt', 'log2'])
    
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    (toolbox.attr_n_estimators, 
                     toolbox.attr_max_depth,
                     toolbox.attr_learning_rate,
                     toolbox.attr_hidden_layers,
                     toolbox.attr_alpha,
                     toolbox.attr_max_features), n=1)
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return toolbox

def evaluate_individual(individual, X, y):
    try:
        n_estimators, max_depth, learning_rate, hidden_layers, alpha, max_features = individual
        
        rf_model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            random_state=42,
            class_weight='balanced'
        )
        
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(int(hidden_layers),),
            learning_rate_init=learning_rate,
            alpha=alpha,
            max_iter=2000,
            random_state=42
        )
        
        gbm_model = GradientBoostingClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            max_features=max_features,
            random_state=42
        )
        
        rf_score = np.mean(cross_val_score(rf_model, X, y, cv=3, scoring='f1_weighted'))
        mlp_score = np.mean(cross_val_score(mlp_model, X, y, cv=3, scoring='f1_weighted'))
        gbm_score = np.mean(cross_val_score(gbm_model, X, y, cv=3, scoring='f1_weighted'))
        
        return max(rf_score, mlp_score, gbm_score),
    except Exception as e:
        logger.error(f"Error evaluating individual: {e}")
        return 0.0,

# === AI Brain Training ===

async def train_ai_brain(chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            df = pd.read_sql_query("SELECT * FROM signals WHERE feedback IS NOT NULL", conn)
        
        if df.empty or len(df[df['feedback'].isin(['win', 'loss'])]) < MIN_FEEDBACK_FOR_TRAINING:
            logger.info("Not enough feedback data to train the AI model yet.")
            return

        df_feedback = df[df['feedback'].isin(['win', 'loss'])].copy()

        if df_feedback.empty:
            logger.info("No 'win' or 'loss' feedback entries to train the AI model.")
            return

        df_feedback['true_action'] = df_feedback.apply(
            lambda row: row['action_for_db'] if row['feedback'] == 'win' else ('SELL' if row['action_for_db'] == 'BUY' else 'BUY'),
            axis=1
        )
        
        y_train_adjusted = df_feedback['true_action']
        X_train_adjusted = df_feedback[[
            'rsi', 'ema', 'ma', 'resistance', 'support',
            'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'atr',
            'adx', 'hma', 't3'
        ]]

        combined_df = pd.concat([X_train_adjusted, y_train_adjusted], axis=1)
        original_size = len(combined_df)
        combined_df.dropna(inplace=True)
        filtered_size = len(combined_df)
        
        if original_size > filtered_size:
            logger.warning(f"Dropped {original_size - filtered_size} rows due to NaN values before training.")
        
        X_train_adjusted = combined_df.drop('true_action', axis=1)
        y_train_adjusted = combined_df['true_action']

        if X_train_adjusted.empty:
            logger.info("No sufficient data after feedback processing and cleaning to train the AI model.")
            await context.bot.send_message(chat_id, "⚠️ AI training skipped: Data was cleaned and no valid entries remained.")
            return

        unique_classes = y_train_adjusted.unique()
        if len(unique_classes) < 2:
            logger.warning(f"Not enough unique classes in feedback data for training. Skipping training.")
            await context.bot.send_message(chat_id, "⚠️ AI training skipped: Not enough diverse feedback.")
            return

        class_counts = y_train_adjusted.value_counts()
        if any(count < 3 for count in class_counts):
            logger.warning(f"Not enough samples per class for 3-fold cross-validation. Skipping training.")
            await context.bot.send_message(chat_id, "⚠️ AI training skipped: Not enough feedback for robust training.")
            return

        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_adjusted, y_train_adjusted)
        
        # Genetic Optimization setup
        toolbox = setup_genetic_algorithm()
        toolbox.register("evaluate", evaluate_individual, X=X_resampled, y=y_resampled)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        population = toolbox.population(n=20)
        algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.3, ngen=10, verbose=False)
        
        best_individual = tools.selBest(population, k=1)[0]
        best_score = evaluate_individual(best_individual, X_resampled, y_resampled)[0]
        
        n_estimators, max_depth, learning_rate, hidden_layers, alpha, max_features = best_individual
        
        rf_model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            random_state=42,
            class_weight='balanced'
        )
        
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(int(hidden_layers),), 
            learning_rate_init=learning_rate,
            alpha=alpha,
            max_iter=2000,
            random_state=42
        )
        
        gbm_model = GradientBoostingClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            max_features=max_features,
            random_state=42
        )
        
        rf_model.fit(X_resampled, y_resampled)
        mlp_model.fit(X_resampled, y_resampled)
        gbm_model.fit(X_resampled, y_resampled)
        
        rf_score = np.mean(cross_val_score(rf_model, X_resampled, y_resampled, cv=3, scoring='f1_weighted'))
        mlp_score = np.mean(cross_val_score(mlp_model, X_resampled, y_resampled, cv=3, scoring='f1_weighted'))
        gbm_score = np.mean(cross_val_score(gbm_model, X_resampled, y_resampled, cv=3, scoring='f1_weighted'))
        
        model_options = [
            (rf_model, "RandomForest", rf_score),
            (mlp_model, "NeuralNetwork", mlp_score),
            (gbm_model, "GradientBoosting", gbm_score)
        ]
        
        model_options.sort(key=lambda x: x[2], reverse=True)
        best_model, model_type, best_accuracy = model_options[0]
        
        model_data = {
            'model': best_model,
            'type': model_type,
            'accuracy': best_accuracy
        }
        joblib.dump(model_data, MODEL_FILE)
        
        logger.info(f"AI model trained. Type: {model_type}, F1 Score: {best_accuracy:.2f}")
        
        await context.bot.send_message(
            chat_id,
            f"🧠 YSBONG TRADER™ AI Brain upgraded!\n"
            f"🤖 Model: {model_type}\n"
            f"🎯 F1 Score: {best_accuracy*100:.1f}%\n"
            f"⚙️ Optimized with Genetic Algorithm\n"
            f"🧑‍🔧Trained on {len(X_resampled)} samples"
        )

    except Exception as e:
        logger.error(f"General error during AI training: {e}", exc_info=True) 
        await context.bot.send_message(chat_id, f"❌ AI training error: {str(e)}")

# === Telegram Handlers ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
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

    user_data[user_id] = {}
    usage_count[user_id] = usage_count.get(user_id, 0)
    
    api_key_from_db = load_saved_keys().get(str(user_id))

    if api_key_from_db:
        user_data[user_id]["api_key"] = api_key_from_db
        kb = []
        for i in range(0, len(PAIRS), 3): 
            row_buttons = [InlineKeyboardButton(get_flagged_pair_name(PAIRS[j]), callback_data=f"pair|{PAIRS[j]}") 
                           for j in range(i, min(i+3, len(PAIRS)))]
            kb.append(row_buttons)

        await update.message.reply_text("🔑 API key loaded.\n💱 Choose Pair:", reply_markup=InlineKeyboardMarkup(kb))
        return

    kb = [[InlineKeyboardButton("✅ I Understand", callback_data="agree_disclaimer")]]
    await update.message.reply_text(
        "⚠️ DISCLAIMER\nThis bot provides educational signals only.\nYou are the engine of your prosperity.",
        reply_markup=InlineKeyboardMarkup(kb)
    )

async def check_joined_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
            
            user_data[user_id] = {}
            usage_count[user_id] = usage_count.get(user_id, 0)
            
            api_key_from_db = load_saved_keys().get(str(user_id))

            if api_key_from_db:
                user_data[user_id]["api_key"] = api_key_from_db
                kb = []
                for i in range(0, len(PAIRS), 3): 
                    row_buttons = [InlineKeyboardButton(get_flagged_pair_name(PAIRS[j]), callback_data=f"pair|{PAIRS[j]}") 
                                for j in range(i, min(i+3, len(PAIRS)))]
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
    reminder = await get_friendly_reminder()
    await update.message.reply_text(reminder, parse_mode='Markdown')

async def get_friendly_reminder() -> str:
    return (
        "📌 *Welcome to YSBONG TRADER™--with an AI BRAIN – Friendly Reminder* 💬\n\n"
        "Hello Trader 👋\n\n"
        "Here’s how to get started with your *real live signals* (not simulation or OTC):\n\n"
        "🤔 *How to Use the Bot*\n"
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
        "🧠 Results depend on live charts, not paper trades.\n\n"
        "⚠️ *No trading on weekends* - the market is closed for non-OTC assets.\n"
        "🧑‍💻 *Beginners:*\n"
        "🙇 Practice first — observe signals.\n"
        "👉 Register here: https://pocket-friends.com/r/w2enb3tukw\n"
        "💵 Deposit when you're confident (min $10).\n\n"
        
        " 🔑 *About TwelveData API Key*\n" 

        "YSBONG TRADER™ uses real-time market data powered by [TwelveData](https://twelvedata.com).\n"
        "You’ll need an API key to activate signals.\n"
        "🆓 **Free Tier (Default when you register)** \n"
        "- ⏱️ Up to 800 API calls per day\n"
        "- 🔄 Max 8 requests per minute\n\n"

        "✌️✌️ GOOD LUCK TRADER ✌️✌️\n\n"

        "🥰 *Be patient. Be disciplined.*\n"
        "😋 *Greedy traders don't last-the market eats them alive.*\n"
        "Respect the market.\n"
        "– *YSBONG TRADER™ powered by PROSPERITY ENGINES™* 💪"
    )

async def disclaimer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    disclaimer_msg = (
    "⚠️ *Financial Risk Disclaimer*\n\n"
    "Trading involves real risk. This bot provides educational signals only.\n"
    "*Not financial advice.*\n\n"
    "🤔 Be wise. Only trade what you can afford to lose.\n"
    "🎯 Results depend on your discipline, not predictions.\n\n"
    "🚫 *Avoid overtrading!* More trades don’t mean more profits — they usually mean more mistakes.\n"
    "⏳🤚 Wait for clean setups, and trust the process.\n"
)
    await update.message.reply_text(disclaimer_msg, parse_mode='Markdown')

async def handle_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
    user_id = update.effective_user.id
    text = update.message.text.strip()
    chat_id = update.effective_chat.id

    if user_data.get(user_id, {}).get("step") == "awaiting_api":
        user_data[user_id]["api_key"] = text
        user_data[user_id]["step"] = None
        save_keys(user_id, text)
        kb = []
        for i in range(0, len(PAIRS), 3): 
            row_buttons = [InlineKeyboardButton(get_flagged_pair_name(PAIRS[j]), callback_data=f"pair|{PAIRS[j]}") 
                           for j in range(i, min(i + 3, len(PAIRS)))]
            kb.append(row_buttons)
        await context.bot.send_message(chat_id, "🔐 API Key saved.\n💱 Choose Currency Pair:", reply_markup=InlineKeyboardMarkup(kb))

async def generate_signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

    loading_msg = await context.bot.send_message(chat_id=chat_id, text="⏳ Analyzing market data...")
    
    memory_candles = get_candle_memory(pair, tf)
    
    status, new_candles = fetch_data(api_key, pair)
    if status == "error":
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
        update_candle_memory(pair, tf, new_candles)
    
    if not candles_to_use:
        await loading_msg.edit_text(f"⚠️ No market data available for {pair} on {tf}. The market might be closed or data is unavailable.")
        return

    indicators = calculate_indicators(candles_to_use)
    
    if not indicators:
        await loading_msg.edit_text(f"❌ Could not calculate indicators for {pair}. Insufficient or malformed data.")
        return

    current_price = float(candles_to_use[-1]["close"])

    action = ""
    confidence = 0.0
    action_for_db = ""
    ai_status_message = ""

    try:
        if os.path.exists(MODEL_FILE):
            model_data = joblib.load(MODEL_FILE)
            model = model_data['model']
            model_type = model_data.get('type', 'RandomForest')
            
            current_features = [
                indicators['RSI'], indicators['EMA'], indicators['MA'],
                indicators['Resistance'], indicators['Support'],
                indicators['MACD'], indicators['MACD_Signal'],
                indicators['Stoch_K'], indicators['Stoch_D'], indicators['ATR'],
                indicators['ADX'], indicators['HMA'], indicators['T3']
            ]
            
            predict_df = pd.DataFrame([current_features], 
                                       columns=[
                                           'rsi', 'ema', 'ma', 'resistance', 'support',
                                           'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'atr',
                                           'adx', 'hma', 't3'
                                       ])

            if model_type == "NeuralNetwork":
                probabilities = model.predict_proba(predict_df)[0]
                classes = model.classes_
            elif model_type == "GradientBoosting":
                probabilities = model.predict_proba(predict_df)[0]
                classes = model.classes_
            else:
                probabilities = model.predict_proba(predict_df)[0]
                classes = model.classes_

            prob_buy = 0.0
            prob_sell = 0.0

            for i, cls in enumerate(classes):
                if cls == 'BUY':
                    prob_buy = probabilities[i]
                elif cls == 'SELL':
                    prob_sell = probabilities[i]

            confidence_threshold = 0.85  # Higher confidence threshold

            if prob_buy >= confidence_threshold and prob_buy >= prob_sell:
                action = "BUY BUY BUY 👆👆👆"
                confidence = prob_buy
                action_for_db = "BUY"
                ai_status_message = f"*(AI: {model_type}, Confidence: {confidence*100:.1f}%)*"
            elif prob_sell >= confidence_threshold and prob_sell > prob_buy:
                action = "SELL SELL SELL 👇👇👇"
                confidence = prob_sell
                action_for_db = "SELL"
                ai_status_message = f"*(AI: {model_type}, Confidence: {confidence*100:.1f}%)*"
            else:
                # Fallback to rule-based with trend confirmation
                trend = indicators['TrendBias']
                adx = indicators['ADX']
                rsi = indicators['RSI']
                stoch_k = indicators['Stoch_K']
                macd_line = indicators['MACD']
                macd_signal = indicators['MACD_Signal']
                
                if trend == "uptrend" and adx >= 20 and rsi > 50 and stoch_k > 50 and macd_line > macd_signal:
                    action = "BUY BUY BUY 👆👆👆"
                    action_for_db = "BUY"
                    ai_status_message = "*(Rule-Based - Strong Uptrend)*"
                elif trend == "downtrend" and adx >= 20 and rsi < 50 and stoch_k < 50 and macd_line < macd_signal:
                    action = "SELL SELL SELL 👇👇👇"
                    action_for_db = "SELL"
                    ai_status_message = "*(Rule-Based - Strong Downtrend)*"
                else:
                    # Price action fallback
                    if current_price > indicators['EMA']:
                        action = "BUY BUY BUY 👆👆👆"
                        action_for_db = "BUY"
                        ai_status_message = "*(Price Action - Above EMA)*"
                    else:
                        action = "SELL SELL SELL 👇👇👇"
                        action_for_db = "SELL"
                        ai_status_message = "*(Price Action - Below EMA)*"
        else:
            logger.warning("AI Model file not found. Using rule-based mode.")
            trend = indicators['TrendBias']
            adx = indicators['ADX']
            rsi = indicators['RSI']
            stoch_k = indicators['Stoch_K']
            macd_line = indicators['MACD']
            macd_signal = indicators['MACD_Signal']
            
            if trend == "uptrend" and adx >= 20 and rsi > 50 and stoch_k > 50 and macd_line > macd_signal:
                action = "BUY BUY BUY 👆👆👆"
                action_for_db = "BUY"
                ai_status_message = "*(Rule-Based - Strong Uptrend)*"
            elif trend == "downtrend" and adx >= 20 and rsi < 50 and stoch_k < 50 and macd_line < macd_signal:
                action = "SELL SELL SELL 👇👇👇"
                action_for_db = "SELL"
                ai_status_message = "*(Rule-Based - Strong Downtrend)*"
            else:
                if current_price > indicators['EMA']:
                    action = "BUY BUY BUY 👆👆👆"
                    action_for_db = "BUY"
                    ai_status_message = "*(Price Action - Above EMA)*"
                else:
                    action = "SELL SELL SELL 👇👇👇"
                    action_for_db = "SELL"
                    ai_status_message = "*(Price Action - Below EMA)*"
    except FileNotFoundError:
        logger.warning("AI Model file not found during prediction. Using rule-based mode.")
        trend = indicators['TrendBias']
        adx = indicators['ADX']
        rsi = indicators['RSI']
        stoch_k = indicators['Stoch_K']
        macd_line = indicators['MACD']
        macd_signal = indicators['MACD_Signal']
        
        if trend == "uptrend" and adx >= 20 and rsi > 50 and stoch_k > 50 and macd_line > macd_signal:
            action = "BUY BUY BUY 👆👆👆"
            action_for_db = "BUY"
            ai_status_message = "*(Rule-Based - Strong Uptrend)*"
        elif trend == "downtrend" and adx >= 20 and rsi < 50 and stoch_k < 50 and macd_line < macd_signal:
            action = "SELL SELL SELL 👇👇👇"
            action_for_db = "SELL"
            ai_status_message = "*(Rule-Based - Strong Downtrend)*"
        else:
            if current_price > indicators['EMA']:
                action = "BUY BUY BUY 👆👆👆"
                action_for_db = "BUY"
                ai_status_message = "*(Price Action - Above EMA)*"
            else:
                action = "SELL SELL SELL 👇👇👇"
                action_for_db = "SELL"
                ai_status_message = "*(Price Action - Below EMA)*"
    except Exception as e:
        logger.error(f"Error during AI prediction: {e}", exc_info=True)
        if current_price > indicators['EMA']:
            action = "BUY BUY BUY 👆👆👆"
            action_for_db = "BUY"
            ai_status_message = "*(Fallback - Above EMA)*"
        else:
            action = "SELL SELL SELL 👇👇👇"
            action_for_db = "SELL"
            ai_status_message = "*(Fallback - Below EMA)*"

    await loading_msg.delete()
    
    signal = (
        f"🥸 *YSBONG TRADER™ AI SIGNAL* 🥸\n\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"💰 *PAIR:* `{get_flagged_pair_name(pair)}`\n"
        f"⏱️ *TIMEFRAME:* `{tf}`\n"
        f"🤗 *ACTION:* **{action}** {ai_status_message}\n"
        f"━━━━━━━━━━━━━━━━━━━\n\n"
        f"📊 *Current Market Data:*\n"
        f"🪙 Price: `{current_price}`\n\n"
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
        f"   • ADX: `{indicators['ADX']}` (Trend Strength)\n"
        f"   • Trend Bias: `{indicators['TrendBias']}`\n"
        f"━━━━━━━━━━━━━━━━━━━\n\n"
        f"💡🫵 *Remember:* Always exercise caution and manage your risk. This is for educational purposes."
    )
    
    feedback_keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🤑 Win", callback_data=f"feedback|win"),
         InlineKeyboardButton("😭 Loss", callback_data=f"feedback|loss")]
    ])
    
    await context.bot.send_message(chat_id=chat_id, text=signal, parse_mode='Markdown', reply_markup=feedback_keyboard)
    
    store_signal(user_id, pair, tf, action_for_db, current_price,
                 indicators["RSI"], indicators["EMA"], indicators["MA"],
                 indicators["Resistance"], indicators["Support"],
                 indicators["MACD"], indicators["MACD_Signal"],
                 indicators["Stoch_K"], indicators["Stoch_D"], indicators["ATR"],
                 indicators["ADX"], indicators["HMA"], indicators["T3"])

def store_signal(user_id: int, pair: str, tf: str, action: str, price: float,
                 rsi: float, ema: float, ma: float, resistance: float, support: float,
                 macd: float, macd_signal: float, stoch_k: float, stoch_d: float, atr: float,
                 adx: float, hma: float, t3: float) -> None:
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO signals (user_id, pair, timeframe, action_for_db, price, rsi, ema, ma, resistance, support, macd, macd_signal, stoch_k, stoch_d, atr, adx, hma, t3)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, pair, tf, action, price, rsi, ema, ma, resistance, support, macd, macd_signal, stoch_k, stoch_d, atr, adx, hma, t3))
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error storing signal to DB: {e}")

async def reset_api(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
            await query.edit_message_text(f"✅ Feedback saved: **{feedback_result.upper()}**. Thank you for teaching me. I LOVE YOU😘😘😘!", parse_mode='Markdown')
        except Exception as e:
            logger.warning(f"Could not edit message for feedback for user {user_id}: {e}")
            await context.bot.send_message(chat_id, f"✅ Feedback saved: **{feedback_result.upper()}**. Thank you for teaching me. I LOVE YOU😘😘😘!")

        try:
            with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM signals WHERE feedback IN ('win','loss')")
                count = c.fetchone()[0]
                if count >= MIN_FEEDBACK_FOR_TRAINING and count % FEEDBACK_BATCH_SIZE == 0:
                    await context.bot.send_message(
                        chat_id,
                        f"🧠 Received enough new feedback (wins AND losses). Starting automatic retraining... Please wait...🤗🤗🤗"
                    )
                    await train_ai_brain(chat_id, context)
        except sqlite3.Error as e:
            logger.error(f"Error counting feedback for training: {e}")

async def brain_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM signals WHERE feedback IN ('win','loss')")
            total_feedback = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM signals WHERE feedback = 'win'")
            wins = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM signals WHERE feedback = 'loss'")
            losses = c.fetchone()[0]
    except sqlite3.Error as e:
        logger.error(f"Error getting brain stats: {e}")
        await update.message.reply_text("❌ Error retrieving brain stats.")
        return

    stats_message = (
        f"🧠 *YSBONG TRADER™ Brain Status*\n\n"
        f"💾**Total Memories (Feedbacks):** `{total_feedback}`\n"
        f"  - 🤑 Wins: `{wins}`\n"
        f"  - 😭 Losses: `{losses}`\n\n"
        f"The AI retrains automatically after every `{FEEDBACK_BATCH_SIZE}` new feedbacks (wins + losses)."
    )
    await update.message.reply_text(stats_message, parse_mode='Markdown')

async def force_train(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("🧠 Starting forced training... Please wait...🤗🤗🤗")
    await train_ai_brain(update.effective_chat.id, context)
    
# === New Features ===
INTRO_MESSAGE = """
📢 WELCOME TO YSBONG TRADER™ – AI SIGNAL SCANNER 📡

🧠 Powered by an intelligent learning system that adapts based on real feedback.  
🔥 Designed to guide both beginners and experienced traders through real-time market signals.

📈 What to Expect:
🔄 Auto-generated signals (BUY/SELL)
🕯️ Smart detection from indicators + candle logic
📝 Community-driven AI learning – YOU help train it
⚡Fast, clean, no-hype trading alerts

🎤 Feedback? Use:
/feedback WIN or /feedback LOSS  
→ Your result helps evolve the brain of the bot 🧠

👩‍❤️‍👨 Invite your friends to join:
https://t.me/ProsperityEngines

🤓 Trade smart. Stay focused. Respect the charts.
🐲 Let the BEAST help you sharpen your instincts.

— YSBONG TRADER™  
“BRAIN BUILT. SIGNAL SENT. PROSPERITY LOADED.”
"""

async def intro_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(INTRO_MESSAGE)

def get_all_users() -> List[int]:
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
    users = get_all_users()
    if not users:
        logger.info("No users found to send intro message")
        return
        
    logger.info(f"Attempting to send intro message to {len(users)} users...")
    
    for user_id in users:
        try:
            chat_member = await app.bot.get_chat_member(chat_id=user_id, user_id=app.bot.id)
            if chat_member.status in [ChatMember.KICKED, ChatMember.LEFT]:
                logger.info(f"Skipping intro message for user {user_id}: Bot is blocked or user left chat.")
                continue

            await app.bot.send_message(chat_id=user_id, text=INTRO_MESSAGE)
            logger.info(f"✅ Intro sent to user: {user_id}")
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.warning(f"❌ Failed to send intro to {user_id}: {e}")

# === Start Bot ===
if __name__ == '__main__':
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

    if not TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set. Bot cannot start.")
        print("ERROR: TELEGRAM_BOT_TOKEN environment variable not set. Please set it or add it to a .env file.")
        exit(1)

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("howto", howto))
    app.add_handler(CommandHandler("disclaimer", disclaimer))
    app.add_handler(CommandHandler("resetapikey", reset_api))
    app.add_handler(CommandHandler("brain_stats", brain_stats))
    app.add_handler(CommandHandler("forcetrain", force_train))
    app.add_handler(CommandHandler("intro", intro_command))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_buttons, pattern="^(pair|timeframe|get_signal|agree_disclaimer).*"))
    app.add_handler(CallbackQueryHandler(feedback_callback_handler, pattern=r"^feedback\|(win|loss)$"))
    app.add_handler(CallbackQueryHandler(check_joined_callback, pattern="^check_joined$"))

    scheduler = BackgroundScheduler()
    scheduler.add_job(lambda: asyncio.run(send_intro_to_all_users(app)), 'cron', day_of_week='mon', hour=9)
    scheduler.start()
    logger.info("⏰ Scheduled weekly intro message configured (Mondays at 9 AM)")

    logger.info("✅ YSBONG TRADER™ with AI Brain is LIVE...")
    app.run_polling(drop_pending_updates=True)