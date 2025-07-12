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

# === Channel Membership Requirement ===
CHANNEL_USERNAME = "#ProsperityEngines"  # Replace with your channel username
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

# === Indicator Calculation ===

def calculate_ema(closes, period):
    if len(closes) < period: return 0.0 # Not enough data
    ema_values = []
    k = 2 / (period + 1)
    ema = closes[0] # Initialize with first close
    ema_values.append(ema)
    for price in closes[1:]:
        ema = price * k + ema * (1 - k)
        ema_values.append(ema)
    return ema_values[-1] # Return the latest EMA value

def calculate_rsi(closes, period=14):
    if len(closes) < period + 1: return 50.0
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    
    gains = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]

    avg_gain = sum(gains[-period:]) / period if gains else 0.0001
    avg_loss = sum(losses[-period:]) / period if losses else 0.0001
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_sma(data, window):
    if len(data) < window: return 0.0
    return sum(data[-window:]) / window

def calculate_macd(closes, fast_period=12, slow_period=26, signal_period=9):
    if len(closes) < max(fast_period, slow_period) + signal_period: return 0.0, 0.0

    ema_fast_values = []
    ema_slow_values = []

    k_fast = 2 / (fast_period + 1)
    k_slow = 2 / (slow_period + 1)

    ema_fast = closes[0]
    ema_slow = closes[0]
    
    ema_fast_values.append(ema_fast)
    ema_slow_values.append(ema_slow)

    for price in closes[1:]:
        ema_fast = price * k_fast + ema_fast * (1 - k_fast)
        ema_slow = price * k_slow + ema_slow * (1 - k_slow)
        ema_fast_values.append(ema_fast)
        ema_slow_values.append(ema_slow)

    macd_line_values = [ef - es for ef, es in zip(ema_fast_values, ema_slow_values)]

    if len(macd_line_values) < signal_period: return macd_line_values[-1], macd_line_values[-1] * 0.8
    
    macd_signal_values = []
    k_signal = 2 / (signal_period + 1)
    
    signal_ema = macd_line_values[signal_period-1] 
    macd_signal_values.append(signal_ema)
    
    for i in range(signal_period, len(macd_line_values)):
        signal_ema = macd_line_values[i] * k_signal + signal_ema * (1 - k_signal)
        macd_signal_values.append(signal_ema)

    return macd_line_values[-1], macd_signal_values[-1]


def calculate_stochastic(highs, lows, closes, k_period=14, d_period=3):
    if len(closes) < k_period + d_period: return 50.0, 50.0
    
    percent_k_values = []
    for i in range(k_period - 1, len(closes)):
        period_lows = lows[i - k_period + 1 : i + 1]
        period_highs = highs[i - k_period + 1 : i + 1]
        
        lowest_low = min(period_lows)
        highest_high = max(period_highs)
        
        if (highest_high - lowest_low) == 0:
            percent_k = 50.0
        else:
            percent_k = ((closes[i] - lowest_low) / (highest_high - lowest_low)) * 100
        percent_k_values.append(percent_k)

    if len(percent_k_values) < d_period: return percent_k_values[-1] if percent_k_values else 50.0, 50.0
    
    percent_d = sum(percent_k_values[-d_period:]) / d_period

    return percent_k_values[-1], percent_d

def calculate_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1: return 0.0
    
    true_ranges = []
    for i in range(1, len(closes)):
        prev_close = closes[i-1]
        
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - prev_close)
        tr3 = abs(lows[i] - prev_close)
        true_ranges.append(max(tr1, tr2, tr3))
    
    if len(true_ranges) < period:
        return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0
    else:
        return sum(true_ranges[-period:]) / period


def calculate_indicators(candles):
    if not candles: return None
    closes = [float(c['close']) for c in candles]
    highs = [float(c['high']) for c in candles]
    lows = [float(c['low']) for c in candles]

    MIN_CANDLES_FOR_FULL_INDICATORS = 50

    if len(closes) < MIN_CANDLES_FOR_FULL_INDICATORS:
        logger.warning(f"Not enough candle data ({len(closes)}) for full indicator calculation. Returning defaults.")
        return {
            "MA": closes[-1] if closes else 0.0,
            "EMA": closes[-1] if closes else 0.0,
            "RSI": 50.0, "Resistance": max(highs) if highs else 0.0, "Support": min(lows) if lows else 0.0,
            "MACD": 0.0, "MACD_Signal": 0.0, "Stoch_K": 50.0, "Stoch_D": 50.0, "ATR": 0.0
        }

    macd_line, macd_signal_line = calculate_macd(closes)
    stoch_k, stoch_d = calculate_stochastic(highs, lows, closes)
    atr_val = calculate_atr(highs, lows, closes)
    
    ma_val = calculate_sma(closes, 14) 
    ema_val = calculate_ema(closes, 9)

    return {
        "MA": round(ma_val, 4),
        "EMA": round(ema_val, 4),
        "RSI": round(calculate_rsi(closes), 2),
        "Resistance": round(max(highs), 4),
        "Support": round(min(lows), 4),
        "MACD": round(macd_line, 4),
        "MACD_Signal": round(macd_signal_line, 4),
        "Stoch_K": round(stoch_k, 2),
        "Stoch_D": round(stoch_d, 2),
        "ATR": round(atr_val, 4)
    }

def fetch_data(api_key, symbol):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": "1min", "apikey": api_key, "outputsize": 100} 
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()
        
        if "status" in data and data["status"] == "error":
            message = data.get("message", "Unknown API Error")
            if "daily limit" in message.lower() or "too many requests" in message.lower():
                return "error", "API Rate Limit Exceeded. Please wait and try again later, or consider a premium TwelveData plan."
            elif "invalid api key" in message.lower() or "auth" in message.lower():
                return "error", "Invalid API Key. Please ensure your key is correct."
            else:
                return "error", f"TwelveData API Error: {message}"
        
        return "ok", list(reversed(data.get("values", [])))
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error fetching data for {symbol}: {http_err} - {http_err.response.text}")
        if http_err.response.status_code == 429:
            time.sleep(5)
            return "error", "API Rate Limit Exceeded. Please try again in a few moments."
        return "error", f"HTTP Error: {http_err}. Please check your internet connection or API key."
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"Connection error fetching data for {symbol}: {conn_err}")
        return "error", "Connection Error. Please check your internet connection."
    except requests.exceptions.Timeout as timeout_err:
        logger.error(f"Timeout error fetching data for {symbol}: {timeout_err}")
        return "error", "Request timed out. TwelveData API might be slow or unreachable."
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching data for {symbol}: {e}")
        return "error", f"An unexpected error occurred: {e}"

# === AI BRAIN MODULE ===
async def train_ai_brain(chat_id=None, context: ContextTypes.DEFAULT_TYPE = None):
    logger.info("🧠 AI Brain training initiated...")
    
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            df = pd.read_sql_query("SELECT * FROM signals WHERE feedback IS NOT NULL AND action_for_model != 'HOLD'", conn)
    except sqlite3.Error as e:
        logger.error(f"Error loading data for AI training: {e}")
        if chat_id and context: await context.bot.send_message(chat_id, f"❌ Error loading training data: {e}")
        return

    if len(df) < MIN_FEEDBACK_FOR_TRAINING:
        msg = f"🧠 Need at least {MIN_FEEDBACK_FOR_TRAINING} feedback entries (excluding HOLD) to train. Currently have {len(df)}."
        logger.warning(msg)
        if chat_id and context: await context.bot.send_message(chat_id, msg)
        return

    df['action_encoded'] = df['action_for_model'].apply(lambda x: 1 if x == 'BUY' else 0)
    df['feedback_encoded'] = df['feedback'].apply(lambda x: 1 if x == 'win' else 0)

    features = [
        'rsi', 'ema', 'ma', 'resistance', 'support',
        'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'atr',
        'action_encoded'
    ]
    target = 'feedback_encoded'

    df.dropna(subset=features + [target], inplace=True)

    if df.empty:
        msg = "Insufficient valid data after dropping NaNs to train the AI model."
        logger.warning(msg)
        if chat_id and context: await context.bot.send_message(chat_id, msg)
        return
    
    if len(df['feedback_encoded'].unique()) < 2:
        msg = "Need at least two classes (win/loss) in feedback to train the AI model effectively."
        logger.warning(msg)
        if chat_id and context: await context.bot.send_message(chat_id, msg)
        return

    X = df[features]
    y = df[target]

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError as e:
        msg = f"Error during data splitting for AI training (likely not enough samples for stratification): {e}"
        logger.error(msg)
        if chat_id and context: await context.bot.send_message(chat_id, msg)
        return

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    logger.info(f"🤖 New AI model trained with accuracy: {accuracy:.2f}")
    
    try:
        joblib.dump(model, MODEL_FILE)
    except Exception as e:
        logger.error(f"Error saving AI model: {e}")
        if chat_id and context: await context.bot.send_message(chat_id, f"❌ Error saving trained model: {e}")
        return

    if chat_id and context:
        await context.bot.send_message(
            chat_id,
            f"✅ 🧠 **AI Brain training complete!**\n\n"
            f"📊 Samples used: {len(df)}\n"
            f"🎯 Model Accuracy: *{accuracy*100:.2f}%*\n\n"
            f"The bot is now smarter."
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

async def disclaimer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    disclaimer_msg = (
        "⚠️ *Financial Risk Disclaimer*\n\n"
        "Trading involves real risk. This bot provides educational signals only.\n"
        "*Not financial advice.*\n\n"
        "📊 Be wise. Only trade what you can afford to lose.\n"
        "💡 Results depend on your discipline, not predictions."
    )
    await update.message.reply_text(disclaimer_msg, parse_mode='Markdown')

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
" - 🔄 Max 8 requests per minute\n"

" ✌️✌️ GOOD LUCK TRADER ✌️✌️\n"

        "⏳ *Be patient. Be disciplined.*\n"
        "😋 *Greedy traders don't last-the market eats them alive.*\n"
        "Respect the market.\n"
        "– *YSBONG TRADER™ powered by PROSPERITY ENGINES™* 🦾"
    )

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

    action = "HOLD ⏸️"
    confidence = 0.0
    action_for_db = "HOLD"
    ai_status_message = ""

    try:
        if os.path.exists(MODEL_FILE):
            model = joblib.load(MODEL_FILE)
            
            features_list = [
                indicators['RSI'], indicators['EMA'], indicators['MA'],
                indicators['Resistance'], indicators['Support'],
                indicators['MACD'], indicators['MACD_Signal'],
                indicators['Stoch_K'], indicators['Stoch_D'], indicators['ATR']
            ]

            predict_df = pd.DataFrame([features_list + [1], features_list + [0]], 
                                       columns=[
                                           'rsi', 'ema', 'ma', 'resistance', 'support',
                                           'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'atr',
                                           'action_encoded'
                                       ])

            prob_win_buy = model.predict_proba(predict_df.iloc[[0]])[0][1]
            prob_win_sell = model.predict_proba(predict_df.iloc[[1]])[0][1]

            confidence_threshold = 0.60

            if prob_win_buy > prob_win_sell and prob_win_buy >= confidence_threshold:
                action = "BUY 🔼"
                confidence = prob_win_buy
                action_for_db = "BUY"
                ai_status_message = f"*(Confidence: {confidence*100:.1f}%)*"
            elif prob_win_sell > prob_win_buy and prob_win_sell >= confidence_threshold:
                action = "SELL 🔽"
                confidence = prob_win_sell
                action_for_db = "SELL"
                ai_status_message = f"*(Confidence: {confidence*100:.1f}%)*"
            else:
                action = "HOLD ⏸️"
                action_for_db = "HOLD"
                ai_status_message = "*(AI: No strong signal)*"

        else:
            action = "BUY 🔼" if current_price > indicators["EMA"] and indicators["RSI"] > 50 else "SELL 🔽"
            action_for_db = "BUY" if "BUY" in action else "SELL"
            ai_status_message = "*(Rule-Based - AI not trained)*"
    except FileNotFoundError:
        logger.warning("AI Model file not found. Running in rule-based mode.")
        action = "BUY 🔼" if current_price > indicators["EMA"] and indicators["RSI"] > 50 else "SELL 🔽"
        action_for_db = "BUY" if "BUY" in action else "SELL"
        ai_status_message = "*(Rule-Based - AI not trained)*"
    except Exception as e:
        logger.error(f"Error during AI prediction: {e}")
        action = "HOLD ⏸️"
        action_for_db = "HOLD"
        ai_status_message = "*(AI: Error in prediction or model loading)*"

    await loading_msg.delete()
    
    signal = (
        f"🥸 *YSBONG TRADER™ AI SIGNAL* 🥸\n\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f" 💰*PAIR:* `{pair}`\n"
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
        
        if add_feedback(user_id, feedback_result):
            try:
                await query.edit_message_text(f"✅ Feedback saved: **{feedback_result.upper()}**. Thank you for teaching me. I LOVE YOU😘😘😘!", parse_mode='Markdown')
            except Exception as e:
                logger.warning(f"Could not edit message for feedback: {e}")
                await context.bot.send_message(query.message.chat_id, f"✅ Feedback saved: **{feedback_result.upper()}**. Thank you for teaching me. I LOVE YOU😘😘😘!")


            try:
                with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
                    count_df = pd.read_sql_query("SELECT COUNT(*) FROM signals WHERE feedback IS NOT NULL AND action_for_model != 'HOLD'", conn)
                    count = count_df.iloc[0,0] if not count_df.empty else 0
            except sqlite3.Error as e:
                logger.error(f"Error counting feedback for retraining trigger: {e}")
                count = 0

            if count >= MIN_FEEDBACK_FOR_TRAINING and count % FEEDBACK_BATCH_SIZE == 0:
                await context.bot.send_message(query.message.chat_id, f"🧠 Received enough new feedback. Starting automatic retraining...")
                await train_ai_brain(query.message.chat_id, context)
        else:
            await query.edit_message_text("🤔 No recent signal found to apply feedback to, or feedback already provided. Please generate a signal first.")


def add_feedback(user_id, feedback):
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute('''
                UPDATE signals
                SET feedback = ?
                WHERE id = (SELECT id FROM signals WHERE user_id = ? AND feedback IS NULL ORDER BY timestamp DESC LIMIT 1)
            ''', (feedback, user_id))
            
            changes = conn.total_changes
            conn.commit()
            return changes > 0
    except sqlite3.Error as e:
        logger.error(f"Error adding feedback to DB: {e}")
        return False

async def brain_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    if not os.path.exists(MODEL_FILE):
        await context.bot.send_message(chat_id, "🧠 The AI Brain has not been trained yet. Please provide feedback on trades to begin the learning process.")
        return

    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            df = pd.read_sql_query("SELECT * FROM signals WHERE feedback IS NOT NULL AND action_for_model != 'HOLD'", conn)
    except sqlite3.Error as e:
        logger.error(f"Error loading data for brain stats: {e}")
        await context.bot.send_message(chat_id, f"❌ Error retrieving brain statistics: {e}")
        return

    total_feedback = len(df)
    if total_feedback < MIN_FEEDBACK_FOR_TRAINING:
         await context.bot.send_message(chat_id, f"🧠 Learning in progress. {total_feedback}/{MIN_FEEDBACK_FOR_TRAINING} feedback entries collected. More data is needed to build the first model.")
         return

    wins = len(df[df['feedback'] == 'win'])
    losses = len(df[df['feedback'] == 'loss'])

    df['action_encoded'] = df['action_for_model'].apply(lambda x: 1 if x == 'BUY' else 0)
    df['feedback_encoded'] = df['feedback'].apply(lambda x: 1 if x == 'win' else 0)
    
    features = [
        'rsi', 'ema', 'ma', 'resistance', 'support',
        'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'atr',
        'action_encoded'
    ]

    df.dropna(subset=features + ['feedback_encoded'], inplace=True)
    if df.empty or len(df['feedback_encoded'].unique()) < 2:
        await context.bot.send_message(chat_id, "📊 Not enough valid or diverse feedback data to calculate current model accuracy.")
        return

    try:
        model = joblib.load(MODEL_FILE)
        accuracy = accuracy_score(df['feedback_encoded'], model.predict(df[features]))

        stats_message = (
            f"🤖 *YSBONG TRADER™ Brain Status*\n\n"
            f"🎯 **Current Model Accuracy:** `{accuracy*100:.2f}%`\n"
            f"📚 **Total Memories (Feedbacks):** `{total_feedback}`\n"
            f"  - 🤑 Wins: `{wins}`\n"
            f"  - 🤮 Losses: `{losses}`\n\n"
            f"The AI retrains automatically after every `{FEEDBACK_BATCH_SIZE}` new feedbacks. Keep it up!"
        )
        await context.bot.send_message(chat_id, stats_message, parse_mode='Markdown')
    except FileNotFoundError:
        await context.bot.send_message(chat_id, "🧠 The AI Brain model file is missing. Please provide feedback to train it.")
    except Exception as e:
        logger.error(f"Error calculating brain stats: {e}")
        await context.bot.send_message(chat_id, f"❌ An error occurred while getting brain stats: {e}")


async def force_train(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(update.message.chat_id, "⏳ Manually starting AI brain training...")
    await train_ai_brain(update.message.chat_id, context)

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

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("howto", howto))
    app.add_handler(CommandHandler("disclaimer", disclaimer))
    app.add_handler(CommandHandler("resetapikey", reset_api))
    app.add_handler(CommandHandler("brain_stats", brain_stats))
    app.add_handler(CommandHandler("forcetrain", force_train))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_buttons, pattern="^(pair|timeframe|get_signal|agree_disclaimer).*"))
    app.add_handler(CallbackQueryHandler(feedback_callback_handler, pattern=r"^feedback\|(win|loss)$"))
    app.add_handler(CallbackQueryHandler(check_joined_callback, pattern="^check_joined$"))

    logger.info("✅ YSBONG TRADER™ with AI Brain is LIVE...")
    app.run_polling(drop_pending_updates=True)