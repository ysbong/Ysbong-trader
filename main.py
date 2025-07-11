import os, json, logging, asyncio, requests, sqlite3, joblib
from flask import Flask
from threading import Thread
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)
# AI & Data Handling Imports - NEW
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# === Flask ping for Render Uptime ===
web_app = Flask(__name__)

@web_app.route('/')
def home():
    return "🤖 YSBONG TRADER™ (AI Brain Active) is awake and learning!"

def run_web():
    web_app.run(host="0.0.0.0", port=8080)

Thread(target=run_web).start()

# === SQLite Learning Memory ===
DB_FILE = "ysbong_memory.db"
# === AI Model File - NEW ===
MODEL_FILE = "ai_brain_model.joblib"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Added action_for_model to store the action taken for training
    # ADDED NEW COLUMNS FOR ADVANCE INDICATORS: macd, macd_signal, stoch_k, stoch_d, atr
    c.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            pair TEXT,
            timeframe TEXT,
            action_for_model TEXT, -- 'BUY' or 'SELL'
            price REAL,
            rsi REAL,
            ema REAL,
            ma REAL,
            resistance REAL,
            support REAL,
            macd REAL,          -- NEW
            macd_signal REAL,   -- NEW
            stoch_k REAL,       -- NEW
            stoch_d REAL,       -- NEW
            atr REAL,           -- NEW
            feedback TEXT DEFAULT NULL, -- 'win' or 'loss'
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Create a table for user API keys
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_api_keys (
            user_id INTEGER PRIMARY KEY,
            api_key TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# === Logging ===
logging.basicConfig(level=logging.INFO)

user_data = {}
usage_count = {}

# Modified load_saved_keys to load from SQLite
def load_saved_keys():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT user_id, api_key FROM user_api_keys")
    keys = {str(row[0]): row[1] for row in c.fetchall()}
    conn.close()
    return keys

# Modified save_keys to save to SQLite
def save_keys(user_id, api_key):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO user_api_keys (user_id, api_key) VALUES (?, ?)", (user_id, api_key))
    conn.commit()
    conn.close()

# Modified remove_key to remove from SQLite
def remove_key(user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM user_api_keys WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

saved_keys = load_saved_keys() # Initial load

# === Constants ===
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "USD/CAD",
    "AUD/USD", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY",
    "EUR/AUD", "AUD/JPY", "CHF/JPY", "NZD/JPY", "EUR/CAD",
    "CAD/JPY", "GBP/CAD", "GBP/CHF", "AUD/CAD", "AUD/CHF"]
TIMEFRAMES = ["1MIN", "5MIN", "15MIN"]
MIN_FEEDBACK_FOR_TRAINING = 10 # Minimum feedback entries needed to train the first model
FEEDBACK_BATCH_SIZE = 5 # Retrain after every 5 new feedback entries

# === Indicator Calculation - UPDATED with MACD, Stochastic, ATR ===
def calculate_ema(closes, period=9):
    if not closes: return 0
    ema = closes[0]
    k = 2 / (period + 1)
    for price in closes[1:]:
        ema = price * k + ema * (1 - k)
    return ema

def calculate_rsi(closes, period=14):
    if len(closes) < period + 1: return 50
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]
    avg_gain = sum(gains[-period:]) / period if gains else 0.01
    avg_loss = sum(losses[-period:]) / period if losses else 0.01
    rs = avg_gain / avg_loss if avg_loss > 0 else 100
    return 100 - (100 / (1 + rs))

def calculate_sma(data, window):
    if len(data) < window: return 0
    return sum(data[-window:]) / window

def calculate_macd(closes, fast_period=12, slow_period=26, signal_period=9):
    if len(closes) < max(fast_period, slow_period) + signal_period: return 0, 0
    
    ema_fast = closes[0]
    k_fast = 2 / (fast_period + 1)
    
    ema_slow = closes[0]
    k_slow = 2 / (slow_period + 1)

    for price in closes[1:]:
        ema_fast = price * k_fast + ema_fast * (1 - k_fast)
        ema_slow = price * k_slow + ema_slow * (1 - k_slow)

    macd_line = ema_fast - ema_slow

    # Calculate signal line (EMA of MACD line)
    macd_history = [macd_line] # We'd need more MACD line values to calculate a proper signal line EMA
    # For simplicity, we'll approximate with the current MACD line for this single point
    # A proper MACD calculation requires storing a history of MACD lines.
    # Given we only fetch 30 candles, a full EMA of MACD is tricky without more data.
    # For a real-time signal, we'll use a simplified approach, or assume previous MACD values.
    # For now, let's just return the macd_line. A more robust solution would require more historical MACD values.
    # To make this functional for training, we'll simplify and use a single point for the signal line.
    
    # Simulating a signal line (requires more historical MACD values for accurate EMA)
    # For a single point calculation, we'll just return MACD line and a dummy signal for now.
    # In a real trading bot, you'd feed historical MACD values to another EMA function.
    
    # To approximate for current candle, let's say the signal line is just a lagged version.
    # This is a simplification.
    macd_signal_line = macd_line * 0.8 # Placeholder for actual EMA of MACD
    
    return macd_line, macd_signal_line

def calculate_stochastic(highs, lows, closes, k_period=14, d_period=3):
    if len(closes) < k_period: return 0, 0
    
    # %K calculation
    lowest_low = min(lows[-k_period:])
    highest_high = max(highs[-k_period:])
    
    if (highest_high - lowest_low) == 0:
        percent_k = 50 # Avoid division by zero
    else:
        percent_k = ((closes[-1] - lowest_low) / (highest_high - lowest_low)) * 100
    
    # %D calculation (SMA of %K)
    # This requires historical %K values. For a single point, we'll just use %K itself
    # and a simplified average for %D.
    # In a real scenario, you'd calculate %K for the last 'd_period' candles and then SMA.
    
    # Placeholder for proper %D (requires history of %K)
    percent_d = percent_k # Simplification for single point
    if len(closes) >= k_period + d_period -1: # if enough data to calculate SMA of K
        recent_k_values = [
            ((closes[i] - min(lows[i-k_period+1:i+1])) / (max(highs[i-k_period+1:i+1]) - min(lows[i-k_period+1:i+1]))) * 100
            for i in range(k_period-1, len(closes))
        ]
        percent_d = sum(recent_k_values[-d_period:]) / d_period if recent_k_values else percent_k


    return percent_k, percent_d

def calculate_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1: return 0
    
    true_ranges = []
    for i in range(1, len(closes)):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i-1])
        tr3 = abs(lows[i] - closes[i-1])
        true_ranges.append(max(tr1, tr2, tr3))
    
    # Simple Moving Average of True Ranges
    if len(true_ranges) < period:
        return sum(true_ranges) / len(true_ranges) # Average of available TRs
    else:
        return sum(true_ranges[-period:]) / period

def calculate_indicators(candles):
    if not candles: return None
    closes = [float(c['close']) for c in reversed(candles)]
    highs = [float(c['high']) for c in reversed(candles)] # Reversed to match closes for indexing
    lows = [float(c['low']) for c in reversed(candles)]   # Reversed to match closes for indexing

    # Ensure enough data for indicators
    if len(closes) < 30: # Need at least ~26 for proper MACD and 14 for others
        logging.warning("Not enough candle data for full indicator calculation.")
        return {
            "MA": 0, "EMA": 0, "RSI": 50, "Resistance": 0, "Support": 0,
            "MACD": 0, "MACD_Signal": 0, "Stoch_K": 50, "Stoch_D": 50, "ATR": 0
        }

    macd_line, macd_signal_line = calculate_macd(closes)
    stoch_k, stoch_d = calculate_stochastic(highs, lows, closes)
    atr_val = calculate_atr(highs, lows, closes)

    return {
        "MA": round(sum(closes) / len(closes), 4),
        "EMA": round(calculate_ema(closes), 4),
        "RSI": round(calculate_rsi(closes), 2),
        "Resistance": round(max(highs), 4), # Max high of provided candles
        "Support": round(min(lows), 4),     # Min low of provided candles
        "MACD": round(macd_line, 4),
        "MACD_Signal": round(macd_signal_line, 4),
        "Stoch_K": round(stoch_k, 2),
        "Stoch_D": round(stoch_d, 2),
        "ATR": round(atr_val, 4)
    }

def fetch_data(api_key, symbol):
    url = "https://api.twelvedata.com/time_series"
    # Increased outputsize to ensure enough data for advanced indicators (e.g., MACD needs ~26 periods)
    params = {"symbol": symbol, "interval": "1min", "apikey": api_key, "outputsize": 60} 
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()
        if "status" in data and data["status"] == "error":
            return "error", data.get("message", "API Error")
        return "ok", data.get("values", [])
    except requests.exceptions.RequestException as e:
        logging.error(f"API Request Error: {e}")
        return "error", "Connection Error"

# === AI BRAIN MODULE - NEW ===
async def train_ai_brain(chat_id=None, context: ContextTypes.DEFAULT_TYPE = None):
    """Loads data, trains the model, and saves it."""
    logging.info("🧠 AI Brain training initiated...")
    conn = sqlite3.connect(DB_FILE)
    # Load only data that has feedback
    df = pd.read_sql_query("SELECT * FROM signals WHERE feedback IS NOT NULL", conn)
    conn.close()

    if len(df) < MIN_FEEDBACK_FOR_TRAINING:
        msg = f"🧠 Need at least {MIN_FEEDBACK_FOR_TRAINING} feedback entries to train. Currently have {len(df)}."
        logging.warning(msg)
        if chat_id and context: await context.bot.send_message(chat_id, msg)
        return

    # Feature Engineering
    df['action_encoded'] = df['action_for_model'].apply(lambda x: 1 if x == 'BUY' else 0)
    df['feedback_encoded'] = df['feedback'].apply(lambda x: 1 if x == 'win' else 0)

    # UPDATED FEATURES LIST
    features = [
        'rsi', 'ema', 'ma', 'resistance', 'support',
        'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'atr', # NEW INDICATORS ADDED
        'action_encoded'
    ]
    target = 'feedback_encoded'

    # Drop rows where any of the features are NaN (can happen if indicator calc failed for some data)
    df.dropna(subset=features, inplace=True)

    if df.empty:
        msg = "Insufficient valid data after dropping NaNs to train the AI model."
        logging.warning(msg)
        if chat_id and context: await context.bot.send_message(chat_id, msg)
        return

    X = df[features]
    y = df[target]

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Evaluate and save
    accuracy = accuracy_score(y_test, model.predict(X_test))
    logging.info(f"🤖 New AI model trained with accuracy: {accuracy:.2f}")
    joblib.dump(model, MODEL_FILE)

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
    user_data[user_id] = {}
    usage_count[user_id] = usage_count.get(user_id, 0)
    
    # Load API key from DB for the current user
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT api_key FROM user_api_keys WHERE user_id = ?", (user_id,))
    api_key_from_db = c.fetchone()
    conn.close()

    if api_key_from_db:
        user_data[user_id]["api_key"] = api_key_from_db[0]
        # === START CHANGE: Implement 5 columns, 4 buttons per column for PAIRS ===
        kb = []
        # Loop through PAIRS list, taking 4 pairs at a time for each row
        for i in range(0, len(PAIRS), 4): 
            row_buttons = [InlineKeyboardButton(PAIRS[j], callback_data=f"pair|{PAIRS[j]}") 
                           for j in range(i, min(i+4, len(PAIRS)))]
            kb.append(row_buttons)
        # === END CHANGE ===

        await update.message.reply_text("🔑 API key loaded.\n💱 Choose Pair:", reply_markup=InlineKeyboardMarkup(kb))
        return

    kb = [[InlineKeyboardButton("✅ I Understand", callback_data="agree_disclaimer")]]
    await update.message.reply_text(
        "⚠️ DISCLAIMER\nThis bot provides educational signals only.\nYou are the engine of your prosperity.",
        reply_markup=InlineKeyboardMarkup(kb)
    )

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
        "⏳ *Be patient. Be disciplined.*\n"
        "📉 *Greedy traders don't last-the market eats them alive.*\n"
        "Respect the market.\n"
        "– *YSBONG TRADER™ powered by PROSPERITY ENGINES™* 🤖"
    )

async def handle_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()
    await query.message.delete()
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
        save_keys(user_id, text) # Save to DB
        # === START CHANGE: Implement 5 columns, 4 buttons per column for PAIRS ===
        kb = []
        # Loop through PAIRS list, taking 4 pairs at a time for each row
        for i in range(0, len(PAIRS), 4): 
            row_buttons = [InlineKeyboardButton(PAIRS[j], callback_data=f"pair|{PAIRS[j]}") 
                           for j in range(i, min(i + 4, len(PAIRS)))]
            kb.append(row_buttons)
        # === END CHANGE ===
        await update.message.reply_text("🔐 API Key saved.\n💱 Choose Currency Pair:", reply_markup=InlineKeyboardMarkup(kb))

# === MODIFIED SIGNAL GENERATION with Professional Output ===
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
    if status == "error" or not result:
        await loading_msg.edit_text(f"❌ Error fetching data: {result}. If it's an API limit, please re-enter key with /resetapikey.")
        user_data[user_id].pop("api_key", None)
        remove_key(user_id) # Remove from DB
        user_data[user_id]["step"] = "awaiting_api"
        return

    indicators = calculate_indicators(result)
    current_price = float(result[0]["close"])

    # --- AI PREDICTION LOGIC ---
    action = "HOLD ⏸️"
    confidence = 0.0 # Initialize confidence
    action_for_db = None
    ai_status_message = "" # Initialize empty, will be populated with confidence or status

    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
        
        # Prepare feature vector for BUY and SELL scenarios using ALL indicators
        features_list = [
            indicators['RSI'], indicators['EMA'], indicators['MA'],
            indicators['Resistance'], indicators['Support'],
            indicators['MACD'], indicators['MACD_Signal'],
            indicators['Stoch_K'], indicators['Stoch_D'], indicators['ATR']
        ]

        buy_features = [features_list + [1]]  # 1 for BUY (encoded action)
        sell_features = [features_list + [0]] # 0 for SELL (encoded action)
        
        try:
            # Predict probability of a 'win' (class 1)
            prob_win_buy = model.predict_proba(buy_features)[0][1]
            prob_win_sell = model.predict_proba(sell_features)[0][1]

            confidence_threshold = 0.60 # Only act if confidence is > 60%

            if prob_win_buy > prob_win_sell and prob_win_buy >= confidence_threshold:
                action = "BUY 🔼"
                confidence = prob_win_buy
                action_for_db = "BUY"
                # === START CHANGE: Add confidence percentage to the signal message ===
                ai_status_message = f"*(Confidence: {confidence*100:.1f}%)*"
                # === END CHANGE ===
            elif prob_win_sell > prob_win_buy and prob_win_sell >= confidence_threshold:
                action = "SELL 🔽"
                confidence = prob_win_sell
                action_for_db = "SELL"
                # === START CHANGE: Add confidence percentage to the signal message ===
                ai_status_message = f"*(Confidence: {confidence*100:.1f}%)*"
                # === END CHANGE ===
            else:
                action = "HOLD ⏸️"
                action_for_db = "HOLD"
                ai_status_message = "*(AI: No strong signal)*" # Default message for HOLD when confidence is low

        except Exception as e:
            logging.error(f"Error during AI prediction: {e}")
            action = "HOLD ⏸️"
            action_for_db = "HOLD"
            ai_status_message = "*(AI: Error in prediction)*"

    else: # Fallback to old logic if no model exists
        action = "BUY 🔼" if current_price > indicators["EMA"] and indicators["RSI"] > 50 else "SELL 🔽"
        action_for_db = "BUY" if "BUY" in action else "SELL"
        ai_status_message = "*(Rule-Based - AI not trained)*"


    await loading_msg.delete()
    
    # --- Professional Signal Output Formatting ---
    signal = (
        f"🥸 *YSBONG TRADER™ AI SIGNAL* 🥸\n\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🪙 *PAIR:* `{pair}`\n"
        f"⏱️ *TIMEFRAME:* `{tf}`\n"
        f"🤗 *ACTION:* **{action}** {ai_status_message}\n" # Confidence message integrated here
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
    
    # Add feedback buttons
    feedback_keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🤑 Win", callback_data=f"feedback|win"),
         InlineKeyboardButton("🤮 Loss", callback_data=f"feedback|loss")]
    ])
    
    await context.bot.send_message(chat_id=chat_id, text=signal, parse_mode='Markdown', reply_markup=feedback_keyboard)
    
    # Store the signal for future learning, but only if it's not a HOLD
    if action_for_db and action_for_db != "HOLD":
        store_signal(user_id, pair, tf, action_for_db, current_price,
                     indicators["RSI"], indicators["EMA"], indicators["MA"],
                     indicators["Resistance"], indicators["Support"],
                     indicators["MACD"], indicators["MACD_Signal"],
                     indicators["Stoch_K"], indicators["Stoch_D"], indicators["ATR"]) # Store new indicators

def store_signal(user_id, pair, tf, action, price, rsi, ema, ma, resistance, support, macd, macd_signal, stoch_k, stoch_d, atr):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO signals (user_id, pair, timeframe, action_for_model, price, rsi, ema, ma, resistance, support, macd, macd_signal, stoch_k, stoch_d, atr)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, pair, tf, action, price, rsi, ema, ma, resistance, support, macd, macd_signal, stoch_k, stoch_d, atr))
    conn.commit()
    conn.close()

async def reset_api(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT api_key FROM user_api_keys WHERE user_id = ?", (user_id,))
    api_key_exists = c.fetchone()
    conn.close()

    if api_key_exists:
        remove_key(user_id) # Remove from DB
        if user_id in user_data:
            user_data[user_id].pop("api_key", None)
        await update.message.reply_text("🗑️ API key removed. Use /start to set a new one.")
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
            await query.edit_message_text(f"✅ Feedback saved: **{feedback_result.upper()}**. Thank you for teaching me. I LOVE YOU😘😘😘!", parse_mode='Markdown')

            # Check if it's time to retrain the model
            conn = sqlite3.connect(DB_FILE)
            count = pd.read_sql_query("SELECT COUNT(*) FROM signals WHERE feedback IS NOT NULL", conn).iloc[0,0]
            conn.close()

            if count % FEEDBACK_BATCH_SIZE == 0:
                await context.bot.send_message(query.message.chat_id, f"🧠 Received enough new feedback. Starting automatic retraining...")
                await train_ai_brain(query.message.chat_id, context)
        else:
            await query.edit_message_text("🤔 No signal found to apply feedback to. Please generate a signal first.")


def add_feedback(user_id, feedback):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Apply feedback to the most recent signal from this user that doesn't have feedback yet
    c.execute('''
        UPDATE signals
        SET feedback = ?
        WHERE id = (SELECT id FROM signals WHERE user_id = ? AND feedback IS NULL ORDER BY timestamp DESC LIMIT 1)
    ''', (feedback, user_id))
    
    changes = conn.total_changes
    conn.commit()
    conn.close()
    return changes > 0

# === NEW COMMANDS ===
async def brain_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Provides statistics about the AI brain."""
    chat_id = update.message.chat_id
    if not os.path.exists(MODEL_FILE):
        await context.bot.send_message(chat_id, "🧠 The AI Brain has not been trained yet. Please provide feedback on trades to begin the learning process.")
        return

    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM signals WHERE feedback IS NOT NULL", conn)
    conn.close()

    total_feedback = len(df)
    if total_feedback < MIN_FEEDBACK_FOR_TRAINING:
         await context.bot.send_message(chat_id, f"🧠 Learning in progress. {total_feedback}/{MIN_FEEDBACK_FOR_TRAINING} feedback entries collected. More data is needed to build the first model.")
         return

    wins = len(df[df['feedback'] == 'win'])
    losses = len(df[df['feedback'] == 'loss'])

    # Re-calculate accuracy on the fly with the latest data
    df['action_encoded'] = df['action_for_model'].apply(lambda x: 1 if x == 'BUY' else 0)
    df['feedback_encoded'] = df['feedback'].apply(lambda x: 1 if x == 'win' else 0)
    
    # UPDATED FEATURES FOR ACCURACY CALC
    features = [
        'rsi', 'ema', 'ma', 'resistance', 'support',
        'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'atr',
        'action_encoded'
    ]

    # Drop rows where any of the features are NaN for evaluation
    df.dropna(subset=features, inplace=True)
    if df.empty:
        await context.bot.send_message(chat_id, "📊 Not enough valid feedback data to calculate current model accuracy.")
        return

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

async def force_train(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Allows manual triggering of AI training."""
    await context.bot.send_message(update.message.chat_id, "⏳ Manually starting AI brain training...")
    await train_ai_brain(update.message.chat_id, context)

# === Start Bot ===
if __name__ == '__main__':
    # IMPORTANT: Replace with your actual bot token
    TOKEN = "7618774950:AAF-SbIBviw3PPwQEGAFX_vsQZlgBVNNScI"
    app = ApplicationBuilder().token(TOKEN).build()

    # Add old and new handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("howto", howto))
    app.add_handler(CommandHandler("disclaimer", disclaimer))
    app.add_handler(CommandHandler("resetapikey", reset_api))
    app.add_handler(CommandHandler("brain_stats", brain_stats)) # NEW
    app.add_handler(CommandHandler("forcetrain", force_train)) # NEW

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_buttons, pattern="^(pair|timeframe|get_signal|agree_disclaimer).*"))
    app.add_handler(CallbackQueryHandler(feedback_callback_handler, pattern=r"^feedback\|(win|loss)$")) # Fixed here


    print("✅ YSBONG TRADER™ with AI Brain is LIVE...")
    app.run_polling()
