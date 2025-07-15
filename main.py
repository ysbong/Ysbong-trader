import os, json, logging, asyncio, requests, sqlite3, joblib, time
from flask import Flask
from threading import Thread
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ChatMember
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)
# AI & Data Handling Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

# === Channel Membership Requirement ===
CHANNEL_USERNAME = "@ProsperityEngines"  # Replace with your channel username
CHANNEL_LINK = "https://t.me/ProsperityEngines"  # Replace with your channel link

async def is_user_joined(user_id, bot):
    try:
        member = await bot.get_chat_member(chat_id=CHANNEL_USERNAME, user_id=user_id)
        return member.status in [ChatMember.MEMBER, ChatMember.OWNER, ChatMember.ADMINISTRATOR]
    except Exception as e:
        logging.error(f"Error checking membership: {e}")
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
def home():
    return "🤖 YSBONG TRADER™ (AI Brain Active) is awake and learning!"

# New /health endpoint as requested
@web_app.route("/health")
def health():
    return "✅ YSBONG™ is alive and kicking!", 200

def run_web():
    # Use a port that Render provides, or default to 8080
    port = int(os.environ.get("PORT", 8080))
    # It's better to explicitly set host to "0.0.0.0" for Docker/containerized environments
    web_app.run(host="0.0.0.0", port=port)

# Start the Flask app in a separate thread
Thread(target=run_web).start()

# === SQLite Learning Memory ===
# If you configure a persistent disk on Render, you might want to set this path
# to a mounted directory, e.g., "/data/ysbong_memory.db"
DB_FILE = "ysbong_memory.db"
# === AI Model File ===
MODEL_FILE = "ai_brain_model.joblib"

# Increased timeout for SQLite connections to reduce "database is locked" errors
SQLITE_TIMEOUT = 10.0 # seconds

def init_db():
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    pair TEXT,
                    timeframe TEXT,
                    action_for_model TEXT, -- 'BUY' or 'SELL' or 'HOLD'
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
            conn.commit()
    except sqlite3.Error as e:
        logging.error(f"SQLite initialization error: {e}")

init_db()

# === Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


user_data = {}
usage_count = {}

def load_saved_keys():
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute("SELECT user_id, api_key FROM user_api_keys")
            keys = {str(row[0]): row[1] for row in c.fetchall()}
            return keys
    except sqlite3.Error as e:
        logger.error(f"Error loading API keys from DB: {e}")
        return {}

def save_keys(user_id, api_key):
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO user_api_keys (user_id, api_key) VALUES (?, ?)", (user_id, api_key))
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error saving API key to DB: {e}")

def remove_key(user_id):
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM user_api_keys WHERE user_id = ?", (user_id,))
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error removing API key from DB: {e}")

saved_keys = load_saved_keys() # Initial load

# === Constants ===
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "USD/CAD",
    "AUD/USD", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY",
    "EUR/AUD", "AUD/JPY", "CHF/JPY", "NZD/JPY", "EUR/CAD",
    "CAD/JPY", "GBP/CAD", "GBP/CHF", "AUD/CAD", "AUD/CHF"]
TIMEFRAMES = ["1MIN", "5MIN", "15MIN"]
MIN_FEEDBACK_FOR_TRAINING = 50 # Increased minimum feedback entries needed to train the first model
FEEDBACK_BATCH_SIZE = 5 # Retrain after every 5 new feedback entries

from typing import List, Tuple, Union, Optional

logger = logging.getLogger(__name__)

# === INDICATOR CALCULATION ===

def calculate_ema(closes: List[float], period: int) -> float:
    if len(closes) < period:
        return 0.0
    k = 2 / (period + 1)
    ema = closes[0]
    for price in closes[1:]:
        ema = price * k + ema * (1 - k)
    return ema

def calculate_rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]
    avg_gain = sum(gains[-period:]) / period if gains else 0.0001
    avg_loss = sum(losses[-period:]) / period if losses else 0.0001
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_sma(data: List[float], window: int) -> float:
    if len(data) < window:
        return 0.0
    return sum(data[-window:]) / window

def calculate_macd(closes: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[float, float]:
    if len(closes) < slow_period + signal_period:
        return 0.0, 0.0
    ema_fast, ema_slow = closes[0], closes[0]
    k_fast, k_slow = 2 / (fast_period + 1), 2 / (slow_period + 1)
    macd_line_values = []
    for price in closes:
        ema_fast = price * k_fast + ema_fast * (1 - k_fast)
        ema_slow = price * k_slow + ema_slow * (1 - k_slow)
        macd_line_values.append(ema_fast - ema_slow)
    k_signal = 2 / (signal_period + 1)
    signal_ema = macd_line_values[0]
    macd_signal_values = []
    for val in macd_line_values:
        signal_ema = val * k_signal + signal_ema * (1 - k_signal)
        macd_signal_values.append(signal_ema)
    return macd_line_values[-1], macd_signal_values[-1]

def calculate_stochastic(highs: List[float], lows: List[float], closes: List[float], k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
    if len(closes) < k_period + d_period:
        return 50.0, 50.0
    percent_k_values = []
    for i in range(k_period - 1, len(closes)):
        period_lows = lows[i - k_period + 1: i + 1]
        period_highs = highs[i - k_period + 1: i + 1]
        lowest = min(period_lows)
        highest = max(period_highs)
        percent_k = 50.0 if highest == lowest else ((closes[i] - lowest) / (highest - lowest)) * 100
        percent_k_values.append(percent_k)
    percent_d = sum(percent_k_values[-d_period:]) / d_period
    return percent_k_values[-1], percent_d

def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 0.0
    tr_list = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        tr_list.append(tr)
    return sum(tr_list[-period:]) / period if len(tr_list) >= period else sum(tr_list) / len(tr_list)

def calculate_indicators(candles: List[dict]) -> Optional[dict]:
    if not candles:
        return None
    closes = [float(c['close']) for c in candles]
    highs = [float(c['high']) for c in candles]
    lows = [float(c['low']) for c in candles]
    if len(closes) < 50:
        if logger: logger.warning(f"Not enough data ({len(closes)} candles). Returning default indicators.")
        return {
            "MA": closes[-1],
            "EMA": closes[-1],
            "RSI": 50.0,
            "Resistance": max(highs),
            "Support": min(lows),
            "MACD": 0.0,
            "MACD_Signal": 0.0,
            "Stoch_K": 50.0,
            "Stoch_D": 50.0,
            "ATR": 0.0
        }
    return {
        "MA": round(calculate_sma(closes, 14), 4),
        "EMA": round(calculate_ema(closes, 9), 4),
        "RSI": round(calculate_rsi(closes), 2),
        "Resistance": round(max(highs), 4),
        "Support": round(min(lows), 4),
        "MACD": round(calculate_macd(closes)[0], 4),
        "MACD_Signal": round(calculate_macd(closes)[1], 4),
        "Stoch_K": round(calculate_stochastic(highs, lows, closes)[0], 2),
        "Stoch_D": round(calculate_stochastic(highs, lows, closes)[1], 2),
        "ATR": round(calculate_atr(highs, lows, closes), 4)
    }

def fetch_data(api_key: str, symbol: str) -> Tuple[str, Union[str, List[dict]]]:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1min",
        "apikey": api_key,
        "outputsize": 100
    }
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()
        if data.get("status") == "error":
            msg = data.get("message", "Unknown API error")
            if "limit" in msg.lower() or "requests" in msg.lower():
                return "error", "API Rate Limit Exceeded. Consider premium plan."
            elif "invalid" in msg.lower() or "auth" in msg.lower():
                return "error", "Invalid API Key."
            return "error", f"TwelveData Error: {msg}"
        return "ok", list(reversed(data.get("values", [])))
    except requests.exceptions.HTTPError as http_err:
        if logger: logger.error(f"HTTP error: {http_err}")
        if http_err.response.status_code == 429:
            time.sleep(5)
            return "error", "API Rate Limit. Try again shortly."
        return "error", f"HTTP Error: {http_err}"
    except requests.exceptions.ConnectionError as conn_err:
        if logger: logger.error(f"Connection error: {conn_err}")
        return "error", "Connection Error. Check internet."
    except requests.exceptions.Timeout as timeout_err:
        if logger: logger.error(f"Timeout: {timeout_err}")
        return "error", "Request timed out."
    except Exception as e:
        if logger: logger.error(f"Unexpected error: {e}")
        return "error", f"Unexpected Error: {e}"

# === TREND & SIGNAL LOGIC (Add-on Section) ===

def detect_trend(closes: List[float]) -> str:
    if len(closes) < 30:
        return "unknown"
    ema9 = calculate_ema(closes, 9)
    ema21 = calculate_ema(closes, 21)
    current_price = closes[-1]
    rsi = calculate_rsi(closes)
    if ema9 > ema21 and current_price > ema9 and rsi > 50:
        return "uptrend"
    elif ema9 < ema21 and current_price < ema9 and rsi < 50:
        return "downtrend"
    else:
        return "sideways"

def get_signal(indicators: dict, trend: str) -> str:
    rsi = indicators['RSI']
    macd = indicators['MACD']
    macd_signal = indicators['MACD_Signal']
    stoch_k = indicators['Stoch_K']
    stoch_d = indicators['Stoch_D']

    # Debug print para makita mo behavior
    print(f"🧠 Trend: {trend} | RSI: {rsi} | MACD: {macd:.2f}/{macd_signal:.2f} | Stoch: {stoch_k:.1f}/{stoch_d:.1f}")

    if trend == "uptrend":
        if rsi > 55 and macd > macd_signal and stoch_k > stoch_d:
            return "BUY"
    elif trend == "downtrend":
        if rsi < 45 and macd < macd_signal and stoch_k < stoch_d:
            return "SELL"
    return "NO SIGNAL"
# === AI BRAIN MODULE ===
async def train_ai_brain(chat_id=None, context: ContextTypes.DEFAULT_TYPE = None):
    logger.info("🧠 AI Brain training initiated...")

    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            # Load both wins and losses for deeper learning
            df = pd.read_sql_query("SELECT * FROM signals WHERE feedback IN ('win', 'loss')", conn)
    except sqlite3.Error as e:
        logger.error(f"Error loading data for AI training: {e}")
        if chat_id and context:
            await context.bot.send_message(chat_id, f"❌ Error loading training data: {e}")
        return

    target = 'action_for_model'  # still BUY or SELL
    features = [
        'rsi', 'ema', 'ma', 'resistance', 'support',
        'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'atr'
    ]

    df.dropna(subset=features + [target], inplace=True)

    if df.empty:
        msg = "❌ Not enough valid data (after dropping NaNs) to train the AI model."
        logger.warning(msg)
        if chat_id and context:
            await context.bot.send_message(chat_id, msg)
        return

    # Make sure both BUY and SELL are present in the data
    if len(df[target].unique()) < 2:
        msg = "❌ AI needs both BUY and SELL examples to train. Currently found: " + str(df[target].unique())
        logger.warning(msg)
        if chat_id and context:
            await context.bot.send_message(chat_id, msg)
        return

    X = df[features]
    y = df[target]  # target action (BUY or SELL)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        msg = f"❌ Stratified split failed (possibly unbalanced data): {e}"
        logger.error(msg)
        if chat_id and context:
            await context.bot.send_message(chat_id, msg)
        return

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    logger.info(f"🤖 AI Model trained with accuracy: {accuracy:.2f}")

    try:
        joblib.dump(model, MODEL_FILE)
    except Exception as e:
        logger.error(f"Error saving AI model: {e}")
        if chat_id and context:
            await context.bot.send_message(chat_id, f"❌ Error saving trained model: {e}")
        return

    if chat_id and context:
        await context.bot.send_message(
            chat_id,
            f"✅ 🧠 **AI Brain training complete!**\n\n"
            f"📊 Samples used: {len(df)} (with WIN + LOSS)\n"
            f"🎯 Accuracy: *{accuracy*100:.2f}%*\n"
            f"🤖 Now smarter based on both wins and losses!"
        )

# === Telegram Handlers ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
        for i in range(0, len(PAIRS), 4): 
            row_buttons = [InlineKeyboardButton(PAIRS[j], callback_data=f"pair|{PAIRS[j]}") 
                           for j in range(i, min(i+4, len(PAIRS)))]
            kb.append(row_buttons)

        await update.message.reply_text("🔑 API key loaded.\n💱 Choose Pair:", reply_markup=InlineKeyboardMarkup(kb))
        return

    kb = [[InlineKeyboardButton("✅ I Understand", callback_data="agree_disclaimer")]]
    await update.message.reply_text(
        "⚠️ DISCLAIMER\nThis bot provides educational signals only.\nYou are the engine of your prosperity.",
        reply_markup=InlineKeyboardMarkup(kb)
    )

async def check_joined_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    chat_id = query.message.chat_id
    await query.answer()

    if query.data == "check_joined":
        if await is_user_joined(user_id, context.bot):
            try:
                await query.message.delete()
            except Exception as e:
                logger.warning(f"Could not delete message: {e}")
            
            # Proceed with normal start flow
            user_data[user_id] = {}
            usage_count[user_id] = usage_count.get(user_id, 0)
            
            api_key_from_db = load_saved_keys().get(str(user_id))

            if api_key_from_db:
                user_data[user_id]["api_key"] = api_key_from_db
                kb = []
                for i in range(0, len(PAIRS), 4): 
                    row_buttons = [InlineKeyboardButton(PAIRS[j], callback_data=f"pair|{PAIRS[j]}") 
                                for j in range(i, min(i+4, len(PAIRS)))]
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

async def howto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reminder = await get_friendly_reminder()
    await update.message.reply_text(reminder, parse_mode='Markdown')

async def get_friendly_reminder():
    return (
        "📌 *Welcome to YSBONG TRADER™--with an AI BRAIN – Friendly Reminder* 💬\n\n"
        "Hello Trader 👋\n\n"
        "Here’s how to get started with your *real live signals* (not simulation or OTC):\n\n"
        "🔧 *How to Use the Bot*\n"
        "1. 🔑 Get your API key from https://twelvedata.com\n"
        "   → Register, log in, dashboard > API Key\n"
        "2. Copy your API KEY || Return to the bot\n"
        "3. Tap the menu button || Tap start\n"
        "4. ✅ Agree to the Disclaimer\n"   
        "   → Paste it here in the bot\n"
        "5. 💱 Choose Trading Pair & Timeframe\n"
        "6. ⚡ Click 📲 GET SIGNAL\n\n"
        "📢 *Note:*\n"
        "🔵 This is not OTC. Signals are based on real market data using your API key.\n"
        "🧠 Results depend on live charts, not paper trades.\n\n"
        "⚠️ *No trading on weekends* - the market is closed for non-OTC assets.\n"
        "🧪 *Beginners:*\n"
        "📚 Practice first — observe signals.\n"
        "👉 Register here: https://pocket-friends.com/r/w2enb3tukw\n"
        "💵 Deposit when you're confident (min $10).\n\n"
        
        " 🔑 *About TwelveData API Key*\n" 

"YSBONG TRADER™ uses real-time market data powered by [TwelveData](https://twelvedata.com).\n"
" You’ll need an API key to activate signals.\n"
" 🆓 **Free Tier (Default when you register)** \n"
" - ⏱️ Up to 800 API calls per day\n"
" - 🔄 Max 8 requests per minute\n\n"

" ✌️✌️ GOOD LUCK TRADER ✌️✌️\n\n"

        "⏳ *Be patient. Be disciplined.*\n"
        "😋 *Greedy traders don't last-the market eats them alive.*\n"
        "Respect the market.\n"
        "– *YSBONG TRADER™ powered by PROSPERITY ENGINES™* 🦾"
    )

async def disclaimer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    disclaimer_msg = (
        "⚠️ *Financial Risk Disclaimer*\n\n"
        "Trading involves real risk. This bot provides educational signals only.\n"
        "*Not financial advice.*\n\n"
        "📊 Be wise. Only trade what you can afford to lose.\n"
        "💡 Results depend on your discipline, not predictions."
    )
    await update.message.reply_text(disclaimer_msg, parse_mode='Markdown')

async def handle_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()
    
    try:
        await query.message.delete()
    except Exception as e:
        logger.warning(f"Could not delete message for user {user_id}: {e}")
        pass

    data = query.data
    if data == "agree_disclaimer":
        await context.bot.send_message(query.message.chat_id, "🔐 Please enter your API key:")
        user_data[user_id] = {"step": "awaiting_api"}
    elif data.startswith("pair|"):
        user_data[user_id]["pair"] = data.split("|")[1]
        kb = [[InlineKeyboardButton(tf, callback_data=f"timeframe|{tf}")] for tf in TIMEFRAMES]
        await context.bot.send_message(query.message.chat_id, "⏰ Choose Timeframe:", reply_markup=InlineKeyboardMarkup(kb))
    elif data.startswith("timeframe|"):
        user_data[user_id]["timeframe"] = data.split("|")[1]
        await context.bot.send_message(
            query.message.chat_id,
            "✅ Ready to generate signal!",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📲 GET SIGNAL", callback_data="get_signal")]])
        )
    elif data == "get_signal":
        await generate_signal(update, context)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()
    if user_data.get(user_id, {}).get("step") == "awaiting_api":
        user_data[user_id]["api_key"] = text
        user_data[user_id]["step"] = None
        save_keys(user_id, text)
        kb = []
        for i in range(0, len(PAIRS), 4): 
            row_buttons = [InlineKeyboardButton(PAIRS[j], callback_data=f"pair|{PAIRS[j]}") 
                           for j in range(i, min(i + 4, len(PAIRS)))]
            kb.append(row_buttons)
        await update.message.reply_text("🔐 API Key saved.\n💱 Choose Currency Pair:", reply_markup=InlineKeyboardMarkup(kb))

async def generate_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    
    status, result = fetch_data(api_key, pair)
    if status == "error":
        await loading_msg.edit_text(f"❌ Error fetching data: {result}. If it's an API limit or invalid key, please use /resetapikey and try again.")
        if "API Key" in result or "rate limit" in result.lower():
            user_data[user_id].pop("api_key", None)
            remove_key(user_id)
            user_data[user_id]["step"] = "awaiting_api"
        return
    
    if not result:
        await loading_msg.edit_text(f"⚠️ No market data available for {pair} on {tf}. The market might be closed or data is unavailable.")
        return

    indicators = calculate_indicators(result)
    
    if not indicators:
        await loading_msg.edit_text(f"❌ Could not calculate indicators for {pair}. Insufficient or malformed data.")
        return

    current_price = float(result[-1]["close"])

    action = ""
    confidence = 0.0
    action_for_db = ""
    ai_status_message = ""

    try:
        if os.path.exists(MODEL_FILE):
            model = joblib.load(MODEL_FILE)
            
            # Prepare features for prediction (only indicators)
            current_features = [
                indicators['RSI'], indicators['EMA'], indicators['MA'],
                indicators['Resistance'], indicators['Support'],
                indicators['MACD'], indicators['MACD_Signal'],
                indicators['Stoch_K'], indicators['Stoch_D'], indicators['ATR']
            ]
            
            predict_df = pd.DataFrame([current_features], 
                                       columns=[
                                           'rsi', 'ema', 'ma', 'resistance', 'support',
                                           'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'atr'
                                       ])

            # Get probabilities for all classes (e.g., 'BUY', 'SELL')
            probabilities = model.predict_proba(predict_df)[0]
            classes = model.classes_ # Get the order of classes from the trained model

            prob_buy = 0.0
            prob_sell = 0.0

            # Map probabilities to 'BUY' and 'SELL' actions
            for i, cls in enumerate(classes):
                if cls == 'BUY':
                    prob_buy = probabilities[i]
                elif cls == 'SELL':
                    prob_sell = probabilities[i]

            confidence_threshold = 0.60 # Threshold for high confidence message

            # Always recommend BUY or SELL based on higher probability
            if prob_buy >= prob_sell: # If equal, lean towards BUY
                action = "BUY 🔼"
                confidence = prob_buy
                action_for_db = "BUY"
            else:
                action = "SELL 🔽"
                confidence = prob_sell
                action_for_db = "SELL"
            
            if confidence >= confidence_threshold:
                ai_status_message = f"*(Confidence: {confidence*100:.1f}%)*"
            else:
                ai_status_message = f"*(AI: Lower confidence: {confidence*100:.1f}%)*"

        else:
            # Fallback if AI model is not trained - always provide BUY/SELL
            if indicators and indicators["RSI"] > 50:
                action = "BUY BUY BUY  🔼🔼🔼"
                action_for_db = "BUY"
            else:
                action = "SELL SELL SELL 🔽🔽🔽"
                action_for_db = "SELL"
            ai_status_message = "*(Rule-Based - AI not trained, very basic logic)*"
    except FileNotFoundError:
        logger.warning("AI Model file not found. Running in rule-based mode (no HOLD).")
        # Fallback if AI model file is missing - always provide BUY/SELL
        if indicators and indicators["RSI"] > 50:
            action = "BUY BUY BUY 🔼🔼🔼"
            action_for_db = "BUY"
        else:
            action = "SELL SELL SELL 🔽🔽🔽"
            action_for_db = "SELL"
        ai_status_message = "*(Rule-Based - AI not trained, very basic logic)*"
    except Exception as e:
        logger.error(f"Error during AI prediction: {e}")
        # On prediction error, always provide BUY/SELL with a note
        if indicators and indicators["RSI"] > 50: # Use RSI if indicators are available
            action = "BUY 🔼"
            action_for_db = "BUY"
        elif indicators: # If RSI is not > 50, and indicators exist
            action = "SELL 🔽"
            action_for_db = "SELL"
        else: # If indicators are not even available, default to BUY
            action = "BUY 🔼"
            action_for_db = "BUY"
        ai_status_message = "*(AI: Error in prediction, using basic logic)*"

    await loading_msg.delete()
    
    signal = (
        f"🥸 *YSBONG TRADER™ AI SIGNAL* 🥸\n\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"💰 *PAIR:* `{pair}`\n"
        f"⏱️ *TIMEFRAME:* `{tf}`\n"
        f"🤗 *ACTION:* **{action}** {ai_status_message}\n"
        f"━━━━━━━━━━━━━━━━━━━\n\n"
        f"📊 *Current Market Data:*\n"
        f"💲 Price: `{current_price}`\n\n"
        f"📈 *Key Indicators:*\n"
        f"   • MA: `{indicators['MA']}`\n"
        f"   • EMA: `{indicators['EMA']}`\n"
        f"   • RSI: `{indicators['RSI']}`\n"
        f"   • Resistance: `{indicators['Resistance']}`\n"
        f"   • Support: `{indicators['Support']}`\n\n"
        f"🚀🦸 *Advanced Indicators:*\n"
        f"   • MACD: `{indicators['MACD']}` (Signal: `{indicators['MACD_Signal']}`)\n"
        f"   • Stoch %K: `{indicators['Stoch_K']}` (Stoch %D: `{indicators['Stoch_D']}`)\n"
        f"   • ATR: `{indicators['ATR']}` (Volatility)\n"
        f"━━━━━━━━━━━━━━━━━━━\n\n"
        f"💡🫵 *Remember:* Always exercise caution and manage your risk. This is for educational purposes."
    )
    
    feedback_keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🤑 Win", callback_data=f"feedback|win"),
         InlineKeyboardButton("🤮 Loss", callback_data=f"feedback|loss")]
    ])
    
    await context.bot.send_message(chat_id=chat_id, text=signal, parse_mode='Markdown', reply_markup=feedback_keyboard)
    
    # Store the signal, which will now always be BUY or SELL
    if action_for_db:
        store_signal(user_id, pair, tf, action_for_db, current_price,
                     indicators["RSI"], indicators["EMA"], indicators["MA"],
                     indicators["Resistance"], indicators["Support"],
                     indicators["MACD"], indicators["MACD_Signal"],
                     indicators["Stoch_K"], indicators["Stoch_D"], indicators["ATR"])

def store_signal(user_id, pair, tf, action, price, rsi, ema, ma, resistance, support, macd, macd_signal, stoch_k, stoch_d, atr):
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO signals (user_id, pair, timeframe, action_for_model, price, rsi, ema, ma, resistance, support, macd, macd_signal, stoch_k, stoch_d, atr)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, pair, tf, action, price, rsi, ema, ma, resistance, support, macd, macd_signal, stoch_k, stoch_d, atr))
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error storing signal to DB: {e}")

async def reset_api(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

async def feedback_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()
    
    data = query.data.split('|')
    if data[0] == "feedback":
        feedback_result = data[1]
        
        # Update the last signal with feedback
        try:
            with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
                c = conn.cursor()
                # Get the last signal ID for this user
                c.execute("SELECT id FROM signals WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1", (user_id,))
                row = c.fetchone()
                if row:
                    signal_id = row[0]
                    c.execute("UPDATE signals SET feedback = ? WHERE id = ?", (feedback_result, signal_id))
                    conn.commit()
                    logger.info(f"Feedback saved for signal {signal_id}: {feedback_result}")
        except sqlite3.Error as e:
            logger.error(f"Error saving feedback: {e}")

        try:
            await query.edit_message_text(f"✅ Feedback saved: **{feedback_result.upper()}**. Thank you for teaching me. I LOVE YOU😘😘😘!", parse_mode='Markdown')
        except Exception as e:
            logger.warning(f"Could not edit message for feedback: {e}")
            await context.bot.send_message(query.message.chat_id, f"✅ Feedback saved: **{feedback_result.upper()}**. Thank you for teaching me. I LOVE YOU😘😘😘!")

        # Trigger retraining if enough feedback
        try:
            with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM signals WHERE feedback IN ('win','loss')")
                count = c.fetchone()[0]
                if count >= MIN_FEEDBACK_FOR_TRAINING and count % FEEDBACK_BATCH_SIZE == 0:
                    await context.bot.send_message(
                        query.message.chat_id,
                        f"🧠 Received enough new feedback (wins AND losses). Starting automatic retraining..."
                    )
                    await train_ai_brain(query.message.chat_id, context)
        except sqlite3.Error as e:
            logger.error(f"Error counting feedback for training: {e}")

async def brain_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

    # Check if model exists and get accuracy? We don't store the accuracy, so we skip.
    stats_message = (
        f"🤖 *YSBONG TRADER™ Brain Status*\n\n"
        f"📚 **Total Memories (Feedbacks):** `{total_feedback}`\n"
        f"  - 🤑 Wins: `{wins}`\n"
        f"  - 🤮 Losses: `{losses}`\n\n"
        f"The AI retrains automatically after every `{FEEDBACK_BATCH_SIZE}` new feedbacks (wins + losses)."
    )
    await update.message.reply_text(stats_message, parse_mode='Markdown')

async def force_train(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await train_ai_brain(update.effective_chat.id, context)
    await update.message.reply_text("🧠 AI training forced! Check back soon for results.")

# === New Features ===
INTRO_MESSAGE = """
📢 WELCOME TO YSBONG TRADER™ – AI SIGNAL SCANNER 📊

🧠 Powered by an intelligent learning system that adapts based on real feedback.  
🔥 Designed to guide both beginners and experienced traders through real-time market signals.

📈 What to Expect:
✅ Auto-generated signals (BUY/SELL)
✅ Smart detection from indicators + candle logic
✅ Community-driven AI learning – YOU help train it
✅ Fast, clean, no-hype trading alerts

💾 Feedback? Use:
/feedback WIN or /feedback LOSS  
→ Your result helps evolve the brain of the bot 🧠

👥 Invite your friends to join:
https://t.me/ProsperityEngines

💡 Trade smart. Stay focused. Respect the charts.
📲 Let the BEAST help you sharpen your instincts.

— YSBONG TRADER™  
“BRAIN BUILT. SIGNAL SENT. PROSPERITY LOADED.”
"""

async def intro_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(INTRO_MESSAGE)

def get_all_users():
    try:
        conn = sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT)
        c = conn.cursor()
        c.execute("SELECT DISTINCT user_id FROM user_api_keys")
        users = [row[0] for row in c.fetchall()]
        conn.close()
        return users
    except sqlite3.Error as e:
        logger.error(f"Error fetching all users: {e}")
        return []

async def send_intro_to_all_users(app):
    users = get_all_users()
    if not users:
        logger.info("No users found to send intro message")
        return
        
    logger.info(f"Sending intro message to {len(users)} users")
    
    for user_id in users:
        try:
            await app.bot.send_message(chat_id=user_id, text=INTRO_MESSAGE)
            logger.info(f"✅ Intro sent to user: {user_id}")
        except Exception as e:
            logger.warning(f"❌ Failed to send intro to {user_id}: {e}")

# === Start Bot ===
if __name__ == '__main__':
    # IMPORTANT: Load token from environment variable
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

    if not TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set. Bot cannot start.")
        # If running locally, you might want to print a more direct message
        # For Pydroid, ensure you've set the variable in its shell or are using .env
        print("ERROR: TELEGRAM_BOT_TOKEN environment variable not set. Please set it or add it to a .env file.")
        exit(1)

    app = ApplicationBuilder().token(TOKEN).build()

    # Add command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("howto", howto))
    app.add_handler(CommandHandler("disclaimer", disclaimer))
    app.add_handler(CommandHandler("resetapikey", reset_api))
    app.add_handler(CommandHandler("brain_stats", brain_stats))
    app.add_handler(CommandHandler("forcetrain", force_train))
    app.add_handler(CommandHandler("intro", intro_command))  # New intro command

    # Add other handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_buttons, pattern="^(pair|timeframe|get_signal|agree_disclaimer).*"))
    app.add_handler(CallbackQueryHandler(feedback_callback_handler, pattern=r"^feedback\|(win|loss)$"))
    app.add_handler(CallbackQueryHandler(check_joined_callback, pattern="^check_joined$"))

    # Setup scheduled intro message
    scheduler = BackgroundScheduler()
    scheduler.add_job(lambda: asyncio.run(send_intro_to_all_users(app)), 'cron', day_of_week='mon', hour=9)
    scheduler.start()
    logger.info("⏰ Scheduled weekly intro message configured (Mondays at 9 AM)")

    logger.info("✅ YSBONG TRADER™ with AI Brain is LIVE...")
    app.run_polling(drop_pending_updates=True)