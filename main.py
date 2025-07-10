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
            feedback TEXT DEFAULT NULL, -- 'win' or 'loss'
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS api_keys (
            user_id INTEGER PRIMARY KEY,
            api_key TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# === Logging ===
logging.basicConfig(level=logging.INFO)

user_data = {}
usage_count = {}

# === SQLite Functions for API Keys ===
def load_api_key_from_db(user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT api_key FROM api_keys WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def save_api_key_to_db(user_id, api_key):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO api_keys (user_id, api_key) VALUES (?, ?)", (user_id, api_key))
    conn.commit()
    conn.close()

def delete_api_key_from_db(user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM api_keys WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

# === Constants ===
PAIRS = ["USD/JPY", "EUR/USD", "GBP/USD", "CAD/JPY", "USD/CAD",
         "AUD/CAD", "GBP/AUD", "EUR/AUD", "GBP/CAD", "CHF/JPY"]
TIMEFRAMES = ["1MIN", "5MIN", "15MIN"]
MIN_FEEDBACK_FOR_TRAINING = 10 # Minimum feedback entries needed to train the first model
FEEDBACK_BATCH_SIZE = 5 # Retrain after every 5 new feedback entries

# === Indicator Calculation (No changes) ===
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

def calculate_indicators(candles):
    if not candles: return None
    closes = [float(c['close']) for c in reversed(candles)]
    highs = [float(c['high']) for c in candles]
    lows = [float(c['low']) for c in candles]
    return {
        "MA": round(sum(closes) / len(closes), 4),
        "EMA": round(calculate_ema(closes), 4),
        "RSI": round(calculate_rsi(closes), 2),
        "Resistance": round(max(highs), 4),
        "Support": round(min(lows), 4)
    }

def fetch_data(api_key, symbol):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": "1min", "apikey": api_key, "outputsize": 30}
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

    features = ['rsi', 'ema', 'ma', 'resistance', 'support', 'action_encoded']
    target = 'feedback_encoded'

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
    
    api_key_from_db = load_api_key_from_db(user_id)

    if api_key_from_db:
        user_data[user_id]["api_key"] = api_key_from_db
        kb = [[InlineKeyboardButton(PAIRS[i], callback_data=f"pair|{PAIRS[i]}"),
               InlineKeyboardButton(PAIRS[i+1], callback_data=f"pair|{PAIRS[i+1]}")]
              for i in range(0, len(PAIRS), 2)]
        kb.append([InlineKeyboardButton("❓ How To Use YSBONG TRADER™", callback_data="show_howto")]) # Add How To button here
        await update.message.reply_text("🔑 API key loaded.\n💱 Choose Pair:", reply_markup=InlineKeyboardMarkup(kb))
    else:
        kb = [[InlineKeyboardButton("✅ I Understand", callback_data="agree_disclaimer")]]
        kb.append([InlineKeyboardButton("❓ How To Use YSBONG TRADER™", callback_data="show_howto")]) # Add How To button here too
        await update.message.reply_text(
            "⚠️ DISCLAIMER\nThis bot provides educational signals only.\nYou are the engine of your prosperity.",
            reply_markup=InlineKeyboardMarkup(kb)
        )

async def howto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reminder = (
        """
# Welcome to YSBONG TRADER™ – Now with an AI Brain! 🧠

Hello Trader 👋

Here’s how to get AI-powered, real-time signals (not simulation or OTC):

## 🚀 Get Started

1.  **Agree to the Disclaimer.**
2.  **Get Your API Key (First Time Only):**
    * Visit [https://twelvedata.com/signup](https://twelvedata.com/signup).
    * Register, log in, and find your API Key on the dashboard.
    * Paste your API key here in the bot.

---

## 📈 Get Signals & Teach the AI

1.  **Choose Your Trading Pair & Timeframe.**
2.  **Click `📲 GET SIGNAL`.**
    * *Note:* Signals are based on real market data using your API key. Results depend on live charts, not paper trades.
3.  **Teach the AI (Crucial!):** After each trade, tell the bot if it was a `win` or `loss`.
    * Use: `/feedback win` or `/feedback loss`.
    * *The more feedback you provide, the smarter the AI becomes!* ✨

---

## 📊 Check AI Status & Beginner Tips

* **Check AI Status:** Use `/brain_stats` to see the AI's current learning progress and accuracy.

* **Beginners:**
    * Practice first — observe signals.
    * Register here: [https://pocket-friends.com/r/w2enb3tukw](https://pocket-friends.com/r/w2enb3tukw)
    * Deposit when you're confident (minimum $10).

---

⏳ **Be patient. Be disciplined.**
📉 **Yesterday's success doesn’t guarantee today’s win.**
Respect the market.

– *YSBONG TRADER™ powered by PROSPERITY ENGINES™* 🤖
"""
    )
    # Determine the chat_id based on whether it's a message update or callback query
    chat_id = update.effective_chat.id 
    await context.bot.send_message(chat_id, reminder, parse_mode='Markdown')

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
    await query.answer() # Always answer the callback query
    await query.message.delete() # Delete the message with the button to keep chat clean
    data = query.data

    if data == "agree_disclaimer":
        await context.bot.send_message(query.message.chat_id, "🔐 Please enter your API key:")
        user_data[user_id] = {"step": "awaiting_api"}
    elif data == "show_howto": # Handle the howto button
        # When a button is pressed, the 'update' object is a CallbackQuery.
        # We need to create a dummy 'Message' update for the howto handler
        # if it expects it, or modify howto to accept CallbackQuery.
        # The easiest is to just call howto and let it handle the effective_chat.id
        await howto(update, context) 
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
        save_api_key_to_db(user_id, text) # Save to SQLite
        kb = [[InlineKeyboardButton(PAIRS[i], callback_data=f"pair|{PAIRS[i]}"),
               InlineKeyboardButton(PAIRS[i+1], callback_data=f"pair|{PAIRS[i+1]}")]
              for i in range(0, len(PAIRS), 2)]
        kb.append([InlineKeyboardButton("❓ How To Use YSBONG TRADER™", callback_data="show_howto")]) # Add How To button here too
        await update.message.reply_text("🔐 API Key saved.\n💱 Choose Currency Pair:", reply_markup=InlineKeyboardMarkup(kb))

# === MODIFIED SIGNAL GENERATION ===
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
        delete_api_key_from_db(user_id) # Delete from SQLite
        user_data[user_id]["step"] = "awaiting_api"
        return

    indicators = calculate_indicators(result)
    current_price = float(result[0]["close"])

    # --- AI PREDICTION LOGIC ---
    action = "HOLD ⏸️"
    confidence = 0
    action_for_db = None

    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
        # Prepare feature vector for BUY and SELL scenarios
        features = [indicators['RSI'], indicators['EMA'], indicators['MA'], indicators['Resistance'], indicators['Support']]
        buy_features = [features + [1]]  # 1 for BUY
        sell_features = [features + [0]] # 0 for SELL
        
        # Predict probability of a 'win' (class 1)
        prob_win_buy = model.predict_proba(buy_features)[0][1]
        prob_win_sell = model.predict_proba(sell_features)[0][1]

        confidence_threshold = 0.60 # Only act if confidence is > 60%

        if prob_win_buy > prob_win_sell and prob_win_buy > confidence_threshold:
            action = f"BUY 🔼 (AI Confidence: {prob_win_buy*100:.1f}%)"
            confidence = prob_win_buy
            action_for_db = "BUY"
        elif prob_win_sell > prob_win_buy and prob_win_sell > confidence_threshold:
            action = f"SELL 🔽 (AI Confidence: {prob_win_sell*100:.1f}%)"
            confidence = prob_win_sell
            action_for_db = "SELL"
        else:
            action = "HOLD ⏸️ (AI: No clear opportunity)"
            action_for_db = "HOLD"

    else: # Fallback to old logic if no model exists
        action = "BUY 🔼" if current_price > indicators["EMA"] and indicators["RSI"] > 50 else "SELL 🔽"
        action += " (Rule-Based)"
        action_for_db = "BUY" if "BUY" in action else "SELL"

    await loading_msg.delete()
    
    signal = (
        f"📡 *YSBONG TRADER™ AI SIGNAL*\n\n"
        f"📍 *PAIR:* {pair}\n"
        f"⏱️ *TIMEFRAME:* {tf}\n"
        f"🤖 *ACTION:* **{action}**\n\n"
        f"— *MARKET DATA* —\n"
        f" attuale Price: {current_price}\n"
        f"MA: {indicators['MA']} | EMA: {indicators['EMA']}\n"
        f"RSI: {indicators['RSI']}\n"
        f"Resistance: {indicators['Resistance']}\n"
        f"Support: {indicators['Support']}"
    )
    
    await context.bot.send_message(chat_id=chat_id, text=signal, parse_mode='Markdown')
    
    # Store the signal for future learning, but only if it's not a HOLD
    if action_for_db and action_for_db != "HOLD":
        store_signal(user_id, pair, tf, action_for_db, current_price,
                     indicators["RSI"], indicators["EMA"], indicators["MA"],
                     indicators["Resistance"], indicators["Support"])

def store_signal(user_id, pair, tf, action, price, rsi, ema, ma, resistance, support):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO signals (user_id, pair, timeframe, action_for_model, price, rsi, ema, ma, resistance, support)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, pair, tf, action, price, rsi, ema, ma, resistance, support))
    conn.commit()
    conn.close()

async def reset_api(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if load_api_key_from_db(user_id):
        delete_api_key_from_db(user_id)
        await update.message.reply_text("🗑️ API key removed. Use /start to set a new one.")
    else:
        await update.message.reply_text("ℹ️ No API key found to reset.")

async def feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args = context.args
    if not args or args[0].lower() not in ["win", "loss"]:
        await update.message.reply_text("❗ Usage: `/feedback win` OR `/feedback loss`")
        return
    
    feedback_result = args[0].lower()
    if add_feedback(user_id, feedback_result):
        await update.message.reply_text(f"✅ Feedback saved: **{feedback_result.upper()}**. Thank you for teaching me!")

        # Check if it's time to retrain the model
        conn = sqlite3.connect(DB_FILE)
        count = pd.read_sql_query("SELECT COUNT(*) FROM signals WHERE feedback IS NOT NULL", conn).iloc[0,0]
        conn.close()

        if count % FEEDBACK_BATCH_SIZE == 0:
            await update.message.reply_text(f"🧠 Received enough new feedback. Starting automatic retraining...")
            await train_ai_brain(update.message.chat_id, context)
    else:
        await update.message.reply_text("🤔 No signal found to apply feedback to. Please generate a signal first.")

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
    features = ['rsi', 'ema', 'ma', 'resistance', 'support', 'action_encoded']
    
    model = joblib.load(MODEL_FILE)
    accuracy = accuracy_score(df['feedback_encoded'], model.predict(df[features]))

    stats_message = (
        f"🤖 *YSBONG TRADER™ Brain Status*\n\n"
        f"🎯 **Current Model Accuracy:** {accuracy*100:.2f}%\n"
        f"📚 **Total Memories (Feedbacks):** {total_feedback}\n"
        f"  - ✅ Wins: {wins}\n"
        f"  - ❌ Losses: {losses}\n\n"
        f"The AI retrains automatically after every {FEEDBACK_BATCH_SIZE} new feedbacks. Keep it up!"
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
    app.add_handler(CommandHandler("feedback", feedback))
    app.add_handler(CommandHandler("brain_stats", brain_stats)) # NEW
    app.add_handler(CommandHandler("forcetrain", force_train)) # NEW

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_buttons))

    print("✅ YSBONG TRADER™ with AI Brain is LIVE...")
    app.run_polling()
