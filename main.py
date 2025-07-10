import os, json, logging, asyncio, requests, sqlite3, joblib, numpy as np
from flask import Flask
from threading import Thread
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from ta import add_all_ta_features  # Technical analysis library

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
MODEL_FILE = "ai_brain_model.joblib"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            pair TEXT,
            timeframe TEXT,
            action_for_model TEXT,
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
            adx REAL,           -- NEW
            bollinger_upper REAL, -- NEW
            bollinger_lower REAL, -- NEW
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
    "CAD/JPY", "GBP/CAD", "GBP/AUD", "AUD/CAD", "AUD/CHF",
 ]
TIMEFRAMES = ["1MIN", "5MIN", "15MIN"]
MIN_FEEDBACK_FOR_TRAINING = 10 # Minimum feedback entries needed to train the first model
FEEDBACK_BATCH_SIZE = 5 # Retrain after every 5 new feedback entries

# === Indicator Calculation ===
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

def calculate_advanced_indicators(candles):
    """Calculate advanced technical indicators"""
    if len(candles) < 26:  # Minimum required for some indicators (e.g., ADX)
        return None
        
    # Create DataFrame with reversed order (oldest first)
    df = pd.DataFrame(candles)[::-1].reset_index(drop=True)
    
    # Convert to numeric
    df['open'] = pd.to_numeric(df['open'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['close'] = pd.to_numeric(df['close'])
    df['volume'] = pd.to_numeric(df['volume'])
    
    # Add all technical analysis features
    df = add_all_ta_features(
        df, 
        open="open", 
        high="high", 
        low="low", 
        close="close", 
        volume="volume"
    )
    
    # Get the latest values
    latest = df.iloc[-1]
    return {
        "MACD": round(latest['momentum_macd'], 4),
        "MACD_Signal": round(latest['momentum_macd_signal'], 4),
        "Stoch_%K": round(latest['momentum_stoch'], 2),
        "Stoch_%D": round(latest['momentum_stoch_signal'], 2),
        "ADX": round(latest['trend_adx'], 2),
        "Bollinger_Upper": round(latest['volatility_bbh'], 4),
        "Bollinger_Lower": round(latest['volatility_bbl'], 4),
    }

def calculate_indicators(candles):
    """Calculate both basic and advanced indicators"""
    if not candles: return None
        
    # Basic indicators
    closes = [float(c['close']) for c in reversed(candles)]
    highs = [float(c['high']) for c in reversed(candles)] # Use reversed for consistent indexing
    lows = [float(c['low']) for c in reversed(candles)]   # Use reversed for consistent indexing
    
    basic_indicators = {
        "MA": round(sum(closes) / len(closes), 4),
        "EMA": round(calculate_ema(closes), 4),
        "RSI": round(calculate_rsi(closes), 2),
        "Resistance": round(max(highs), 4),
        "Support": round(min(lows), 4),
    }
    
    # Advanced indicators
    advanced_indicators = calculate_advanced_indicators(candles)
    
    # Merge dictionaries, prioritize advanced indicators if available
    if advanced_indicators:
        return {**basic_indicators, **advanced_indicators}
    else:
        # Provide default values for advanced indicators if not enough data for TA-Lib
        basic_indicators.update({
            "MACD": 0.0, "MACD_Signal": 0.0,
            "Stoch_%K": 50.0, "Stoch_%D": 50.0,
            "ADX": 0.0,
            "Bollinger_Upper": basic_indicators['MA'], "Bollinger_Lower": basic_indicators['MA']
        })
        return basic_indicators


def fetch_data(api_key, symbol, output_size=100):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": "1min", "apikey": api_key, "outputsize": output_size}
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

# === AI BRAIN MODULE ===
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

    features = [
        'rsi', 'ema', 'ma', 'resistance', 'support', 
        'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'adx',
        'bollinger_upper', 'bollinger_lower', 'action_encoded'
    ]
    target = 'feedback_encoded'

    # Filter out rows where any feature is NaN or infinite before splitting
    df_cleaned = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)

    if len(df_cleaned) == 0:
        msg = "🧠 No valid data after cleaning for AI training. Cannot train model."
        logging.warning(msg)
        if chat_id and context: await context.bot.send_message(chat_id, msg)
        return
        
    X = df_cleaned[features]
    y = df_cleaned[target]

    # Ensure there are enough samples for both classes for stratification
    if len(y.unique()) < 2 or y.value_counts().min() < 2:
        msg = "🧠 Not enough distinct feedback types or samples per type for stratified split. Cannot train robustly."
        logging.warning(msg)
        if chat_id and context: await context.bot.send_message(chat_id, msg)
        # Attempt to train without stratification if not enough samples for it
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    # Train the model
    model = RandomForestClassifier(
        n_estimators=200, 
        random_state=42, 
        class_weight='balanced',
        max_depth=10,
        min_samples_split=5
    )
    model.fit(X_train, y_train)

    # Evaluate and save
    accuracy = accuracy_score(y_test, model.predict(X_test))
    logging.info(f"🤖 New AI model trained with accuracy: {accuracy:.2f}")
    joblib.dump(model, MODEL_FILE)

    if chat_id and context:
        await context.bot.send_message(
            chat_id,
            f"✅ 🧠 **AI Brain training complete!**\n\n"
            f"📊 Samples used: {len(df_cleaned)}\n"
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
        # Create keyboard with 4 buttons per row
        kb = []
        for i in range(0, len(PAIRS), 4):
            row_buttons = [InlineKeyboardButton(PAIRS[j], callback_data=f"pair|{PAIRS[j]}") for j in range(i, min(i+4, len(PAIRS)))]
            kb.append(row_buttons)
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
        "1. ✅ Agree to the Disclaimer\n"
        "2. 🔑 Get your API key from https://twelvedata.com\n"
        "   → Register, login, dashboard > API Key\n"
        "   → Paste it here in the bot\n"
        "3. 💱 Choose Trading Pair & Timeframe\n"
        "4. ⚡ Click 📲 GET SIGNAL\n\n"
        "📢 *Note:*\n"
        "🔵 This is not OTC. Signals are based on real market data using your API key.\n"
        "🧠 Results depend on live charts, not paper trades.\n\n"
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
    # Handle feedback buttons
    elif data.startswith("feedback_"):
        parts = data.split("_")
        if len(parts) >= 2:
            feedback_type = parts[1].lower()
            if feedback_type in ["win", "loss"]:
                await handle_feedback_button(update, context, feedback_type)

async def handle_feedback_button(update: Update, context: ContextTypes.DEFAULT_TYPE, feedback_type: str):
    """Handle feedback from inline buttons"""
    query = update.callback_query
    user_id = query.from_user.id
    chat_id = query.message.chat_id
    await query.answer()
    
    if add_feedback(user_id, feedback_type):
        await query.edit_message_text(text=f"✅ Feedback recorded: {feedback_type.upper()}! Thank you for teaching me!")
        
        # Check if it's time to retrain the model
        conn = sqlite3.connect(DB_FILE)
        count = pd.read_sql_query("SELECT COUNT(*) FROM signals WHERE feedback IS NOT NULL", conn).iloc[0,0]
        conn.close()

        if count >= MIN_FEEDBACK_FOR_TRAINING and count % FEEDBACK_BATCH_SIZE == 0:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"🧠 Received enough new feedback. Starting automatic retraining..."
            )
            await train_ai_brain(chat_id, context)
    else:
        await query.edit_message_text(text="🤔 No signal found to apply feedback to. Please generate a signal first.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()
    if user_data.get(user_id, {}).get("step") == "awaiting_api":
        user_data[user_id]["api_key"] = text
        user_data[user_id]["step"] = None
        save_keys(user_id, text) # Save to DB
        # Create keyboard with 4 buttons per row
        kb = []
        for i in range(0, len(PAIRS), 4):
            row_buttons = [InlineKeyboardButton(PAIRS[j], callback_data=f"pair|{PAIRS[j]}") for j in range(i, min(i+4, len(PAIRS)))]
            kb.append(row_buttons)
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

    loading_msg = await context.bot.send_message(chat_id=chat_id, text="⏳ Analyzing market data with advanced indicators...")
    
    # Fetch more data for advanced indicators
    status, result = fetch_data(api_key, pair, output_size=100)  # Get more data
    if status == "error" or not result:
        await loading_msg.edit_text(f"❌ Error fetching data: {result}")
        return

    # Check if we have enough data for advanced indicators (min 26 for ADX, MACD, etc.)
    if len(result) < 26: 
        await loading_msg.edit_text(f"❌ Error: Only {len(result)} candles received. Need at least 26 for robust advanced indicator calculation. Please try again or check your API key/symbol.")
        return

    indicators = calculate_indicators(result)
    if not indicators:
        await loading_msg.edit_text("❌ Error calculating indicators. Please try again.")
        return

    current_price = float(result[0]["close"])

    # --- ENHANCED AI PREDICTION / FALLBACK LOGIC ---
    action = "HOLD ⏸️"
    confidence = 0.0
    action_for_db = "HOLD"
    signal_source = "Rule-Based" # Default source

    # Prepare feature vector for prediction (even if model doesn't exist yet, for consistency)
    features_vector = [
        indicators['RSI'], indicators['EMA'], indicators['MA'], 
        indicators['Resistance'], indicators['Support'],
        indicators['MACD'], indicators['MACD_Signal'],
        indicators['Stoch_%K'], indicators['Stoch_%D'],
        indicators['ADX'],
        indicators['Bollinger_Upper'], indicators['Bollinger_Lower']
    ]

    if os.path.exists(MODEL_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            
            # Predict probabilities for BUY (1) and SELL (0)
            # We need to append the 'action_encoded' feature for prediction
            buy_features = [features_vector + [1]]  # 1 for BUY
            sell_features = [features_vector + [0]] # 0 for SELL
            
            prob_win_buy = model.predict_proba(buy_features)[0][1]
            prob_win_sell = model.predict_proba(sell_features)[0][1]

            confidence_threshold = 0.70  # Higher threshold for AI accuracy

            if prob_win_buy > prob_win_sell and prob_win_buy > confidence_threshold:
                action = f"BUY 🔼 ⬆️"
                confidence = prob_win_buy
                action_for_db = "BUY"
                signal_source = "AI Brain"
            elif prob_win_sell > prob_win_buy and prob_win_sell > confidence_threshold:
                action = f"SELL 🔽 ⬇️"
                confidence = prob_win_sell
                action_for_db = "SELL"
                signal_source = "AI Brain"
            else:
                # AI is uncertain, fall back to rule-based
                logging.info(f"AI uncertain (Buy: {prob_win_buy:.2f}, Sell: {prob_win_sell:.2f}). Falling back to rule-based.")
                pass # Continue to rule-based logic below
        except Exception as e:
            logging.error(f"Error loading or predicting with AI model: {e}. Falling back to rule-based.")
            pass # Fallback to rule-based logic

    # Rule-based fallback if AI is not used or not confident
    if signal_source == "Rule-Based": # Only apply if AI hasn't given a confident signal
        buy_signals = 0
        sell_signals = 0
        
        # MACD crossover
        if indicators['MACD'] > indicators['MACD_Signal']:
            buy_signals += 1
        elif indicators['MACD'] < indicators['MACD_Signal']:
            sell_signals += 1
            
        # Stochastic (Oversold/Overbought)
        if indicators['Stoch_%K'] < 20 and indicators['Stoch_%D'] < 20:
            buy_signals += 1
        elif indicators['Stoch_%K'] > 80 and indicators['Stoch_%D'] > 80:
            sell_signals += 1
            
        # Bollinger Bands (Price breaking outside)
        if current_price < indicators['Bollinger_Lower']:
            buy_signals += 1
        elif current_price > indicators['Bollinger_Upper']:
            sell_signals += 1
            
        # RSI (Oversold/Overbought)
        if indicators['RSI'] < 30:
            buy_signals += 1
        elif indicators['RSI'] > 70:
            sell_signals += 1
            
        # Simple Moving Average Crossover (current price vs MA)
        if current_price > indicators['MA']:
            buy_signals += 1
        elif current_price < indicators['MA']:
            sell_signals += 1

        if buy_signals > sell_signals and buy_signals >= 3: # Require at least 3 indicators for a strong rule-based signal
            action = "BUY 🔼 ⬆️"
            action_for_db = "BUY"
            confidence = (buy_signals / (buy_signals + sell_signals)) if (buy_signals + sell_signals) > 0 else 0
        elif sell_signals > buy_signals and sell_signals >= 3: # Require at least 3 indicators for a strong rule-based signal
            action = "SELL 🔽 ⬇️"
            action_for_db = "SELL"
            confidence = (sell_signals / (buy_signals + sell_signals)) if (buy_signals + sell_signals) > 0 else 0
        else:
            action = "HOLD ⏸️"
            action_for_db = "HOLD"
            confidence = 0.0 # No strong directional confidence

    # PROFESSIONAL SIGNAL FORMATTING
    signal = (
        f"🚀 *YSBONG TRADER™ ULTRA SIGNAL*\n"
        f"══════════════════════════\n"
        f"💹 *PAIR:* `{pair}`\n"
        f"⏱️ *TIMEFRAME:* `{tf}`\n"
        f"💰 *PRICE:* `{current_price}`\n\n"
        f"⚡ *ACTION:* {'✅' if 'BUY' in action else '❌' if 'SELL' in action else '⚠️'} "
        f"*{action.split()[0]}*\n"
        f"📊 *Confidence:* `{confidence*100:.1f}%` ({signal_source})\n"
        f"══════════════════════════\n"
        f"📈 *TECHNICAL INDICATORS*\n"
        f"├─ RSI: `{indicators['RSI']:.2f}` {'🔴' if indicators['RSI'] > 70 else '🟢' if indicators['RSI'] < 30 else '⚪'}\n"
        f"├─ EMA: `{indicators['EMA']:.4f}`\n"
        f"├─ MA: `{indicators['MA']:.4f}`\n" # Added MA to display
        f"├─ MACD: `{indicators['MACD']:.4f}` | Signal: `{indicators['MACD_Signal']:.4f}`\n"
        f"├─ Stoch: %K=`{indicators['Stoch_%K']:.2f}`, %D=`{indicators['Stoch_%D']:.2f}`\n"
        f"├─ ADX: `{indicators['ADX']:.2f}` {'🟢' if indicators['ADX'] > 25 else '🔴'}\n"
        f"├─ Bollinger: Upper=`{indicators['Bollinger_Upper']:.4f}`, Lower=`{indicators['Bollinger_Lower']:.4f}`\n"
        f"├─ Resistance: `{indicators['Resistance']:.4f}`\n"
        f"└─ Support: `{indicators['Support']:.4f}`\n"
        f"══════════════════════════\n"
        f"📌 _Signal ID: #{np.random.randint(1000,9999)}_"
    )
    
    # Prepare reply markup for feedback buttons
    if action_for_db in ["BUY", "SELL"]: # Only ask for feedback on actionable signals
        reply_markup = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Win", callback_data="feedback_win"),
             InlineKeyboardButton("❌ Loss", callback_data="feedback_loss")]
        ])
    else:
        reply_markup = None
    
    # Store the signal if it's actionable for the model to learn from later
    if action_for_db in ["BUY", "SELL"]:
        store_signal(user_id, pair, tf, action_for_db, current_price,
                     indicators["RSI"], indicators["EMA"], indicators["MA"],
                     indicators["Resistance"], indicators["Support"],
                     indicators["MACD"], indicators["MACD_Signal"],
                     indicators["Stoch_%K"], indicators["Stoch_%D"],
                     indicators["ADX"],
                     indicators["Bollinger_Upper"], indicators["Bollinger_Lower"])

    # Delete loading message and send the final signal
    await loading_msg.delete()
    try:
        await context.bot.send_photo(
            chat_id=chat_id,
            photo=open("signal_bg.jpg", "rb") if os.path.exists("signal_bg.jpg") else None,
            caption=signal,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    except Exception as e:
        logging.error(f"Error sending photo, sending as text instead: {e}")
        await context.bot.send_message(
            chat_id=chat_id,
            text=signal,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )


def store_signal(user_id, pair, tf, action, price, rsi, ema, ma, resistance, support,
                 macd, macd_signal, stoch_k, stoch_d, adx, bollinger_upper, bollinger_lower):
    """Store signal with advanced indicators"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO signals (
            user_id, pair, timeframe, action_for_model, price, 
            rsi, ema, ma, resistance, support,
            macd, macd_signal, stoch_k, stoch_d, adx,
            bollinger_upper, bollinger_lower
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, pair, tf, action, price, rsi, ema, ma, resistance, support,
          macd, macd_signal, stoch_k, stoch_d, adx, bollinger_upper, bollinger_lower))
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

        if count >= MIN_FEEDBACK_FOR_TRAINING and count % FEEDBACK_BATCH_SIZE == 0:
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
    
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM signals WHERE feedback IS NOT NULL", conn)
    conn.close()

    total_feedback = len(df)
    
    if not os.path.exists(MODEL_FILE) or total_feedback < MIN_FEEDBACK_FOR_TRAINING:
        msg = f"🧠 The AI Brain is learning! Currently, it has {total_feedback} feedback entries."
        if total_feedback < MIN_FEEDBACK_FOR_TRAINING:
            msg += f"\nIt needs at least {MIN_FEEDBACK_FOR_TRAINING} feedback entries to build its first model. Keep providing feedback after trades!"
        else:
            msg += "\nHowever, the model file was not found. Please try `/forcetrain`."
        await context.bot.send_message(chat_id, msg)
        return

    wins = len(df[df['feedback'] == 'win'])
    losses = len(df[df['feedback'] == 'loss'])

    # Re-calculate accuracy on the fly with the latest data
    df['action_encoded'] = df['action_for_model'].apply(lambda x: 1 if x == 'BUY' else 0)
    df['feedback_encoded'] = df['feedback'].apply(lambda x: 1 if x == 'win' else 0)
    features = [
        'rsi', 'ema', 'ma', 'resistance', 'support', 
        'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'adx',
        'bollinger_upper', 'bollinger_lower', 'action_encoded'
    ]

    # Filter out rows where any feature is NaN or infinite before predicting
    df_cleaned = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)

    if len(df_cleaned) == 0:
        await context.bot.send_message(chat_id, "🧠 No valid data for calculating brain statistics after cleaning. Cannot provide accuracy.")
        return

    try:
        model = joblib.load(MODEL_FILE)
        predictions = model.predict(df_cleaned[features])
        accuracy = accuracy_score(df_cleaned['feedback_encoded'], predictions)
    except Exception as e:
        accuracy = 0.0 # Default if model fails to load or predict
        logging.error(f"Error calculating accuracy for brain stats: {e}")
        await context.bot.send_message(chat_id, "⚠️ Error retrieving AI model accuracy. Model might be corrupted or data is insufficient.")


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
    app.add_handler(CommandHandler("brain_stats", brain_stats))
    app.add_handler(CommandHandler("forcetrain", force_train))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_buttons))

    print("✅ YSBONG TRADER™ with AI Brain is LIVE...")
    app.run_polling()
