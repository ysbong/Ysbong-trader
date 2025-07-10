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
from ta import add_all_ta_features  # NEW: Technical analysis library

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

# ... [rest of the helper functions remain unchanged] ...

# === Advanced Indicator Calculation ===
def calculate_advanced_indicators(candles):
    """Calculate advanced technical indicators"""
    if len(candles) < 26:  # Minimum required for some indicators
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
    highs = [float(c['high']) for c in candles]
    lows = [float(c['low']) for c in candles]
    
    basic_indicators = {
        "MA": round(sum(closes) / len(closes), 4),
        "EMA": round(calculate_ema(closes), 4),
        "RSI": round(calculate_rsi(closes), 2),
        "Resistance": round(max(highs), 4),
        "Support": round(min(lows), 4),
    }
    
    # Advanced indicators
    advanced_indicators = calculate_advanced_indicators(candles)
    
    return {**basic_indicators, **advanced_indicators}

# ... [rest of the existing functions] ...

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

    indicators = calculate_indicators(result)
    current_price = float(result[0]["close"])

    # --- ENHANCED AI PREDICTION ---
    action = "HOLD ⏸️"
    confidence = 0
    action_for_db = None

    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
        # Prepare feature vector with advanced indicators
        features = [
            indicators['RSI'], indicators['EMA'], indicators['MA'], 
            indicators['Resistance'], indicators['Support'],
            indicators['MACD'], indicators['MACD_Signal'],
            indicators['Stoch_%K'], indicators['Stoch_%D'],
            indicators['ADX'],
            indicators['Bollinger_Upper'], indicators['Bollinger_Lower']
        ]
        
        buy_features = [features + [1]]  # 1 for BUY
        sell_features = [features + [0]] # 0 for SELL
        
        prob_win_buy = model.predict_proba(buy_features)[0][1]
        prob_win_sell = model.predict_proba(sell_features)[0][1]

        confidence_threshold = 0.70  # Higher threshold for accuracy

        if prob_win_buy > prob_win_sell and prob_win_buy > confidence_threshold:
            action = f"BUY 🔼 ⬆️ (AI Confidence: {prob_win_buy*100:.1f}%)"
            confidence = prob_win_buy
            action_for_db = "BUY"
        elif prob_win_sell > prob_win_buy and prob_win_sell > confidence_threshold:
            action = f"SELL 🔽 ⬇️ (AI Confidence: {prob_win_sell*100:.1f}%)"
            confidence = prob_win_sell
            action_for_db = "SELL"
        else:
            action = "HOLD ⏸️ (AI: Market conditions uncertain)"
            action_for_db = "HOLD"

    else:  # Enhanced fallback logic
        buy_signals = 0
        sell_signals = 0
        
        # MACD crossover
        if indicators['MACD'] > indicators['MACD_Signal']:
            buy_signals += 1
        else:
            sell_signals += 1
            
        # Stochastic
        if indicators['Stoch_%K'] < 20:
            buy_signals += 1
        elif indicators['Stoch_%K'] > 80:
            sell_signals += 1
            
        # Bollinger Bands
        if current_price < indicators['Bollinger_Lower']:
            buy_signals += 1
        elif current_price > indicators['Bollinger_Upper']:
            sell_signals += 1
            
        # RSI
        if indicators['RSI'] < 30:
            buy_signals += 1
        elif indicators['RSI'] > 70:
            sell_signals += 1
            
        if buy_signals >= 3:
            action = "BUY 🔼 ⬆️ (Multi-indicator)"
            action_for_db = "BUY"
        elif sell_signals >= 3:
            action = "SELL 🔽 ⬇️ (Multi-indicator)"
            action_for_db = "SELL"
        else:
            action = "HOLD ⏸️ (Rule-Based)"
            action_for_db = "HOLD"

    # PROFESSIONAL SIGNAL FORMATTING
    signal = (
        f"🚀 *YSBONG TRADER™ ULTRA SIGNAL*\n"
        f"══════════════════════════\n"
        f"💹 *PAIR:* `{pair}`\n"
        f"⏱️ *TIMEFRAME:* `{tf}`\n"
        f"💰 *PRICE:* `{current_price}`\n\n"
        f"⚡ *ACTION:* {'✅' if 'BUY' in action else '❌' if 'SELL' in action else '⚠️'} "
        f"*{action.split()[0]}*\n"
        f"📊 *Confidence:* `{confidence*100 if confidence > 0 else 'N/A'}%`\n"
        f"══════════════════════════\n"
        f"📈 *TECHNICAL INDICATORS*\n"
        f"├─ RSI: `{indicators['RSI']}` {'🔴' if indicators['RSI'] > 70 else '🟢' if indicators['RSI'] < 30 else '⚪'}\n"
        f"├─ EMA: `{indicators['EMA']}`\n"
        f"├─ MACD: `{indicators['MACD']}` | Signal: `{indicators['MACD_Signal']}`\n"
        f"├─ Stoch: %K=`{indicators['Stoch_%K']}`, %D=`{indicators['Stoch_%D']}`\n"
        f"├─ ADX: `{indicators['ADX']}` {'🟢' if indicators['ADX'] > 25 else '🔴'}\n"
        f"├─ Bollinger: Upper=`{indicators['Bollinger_Upper']}`, Lower=`{indicators['Bollinger_Lower']}`\n"
        f"├─ Resistance: `{indicators['Resistance']}`\n"
        f"└─ Support: `{indicators['Support']}`\n"
        f"══════════════════════════\n"
        f"📌 _Signal ID: #{np.random.randint(1000,9999)}_"
        f"\n\n🔔 /feedback win|loss"
    )
    
    await context.bot.send_photo(
        chat_id=chat_id,
        photo=open("signal_bg.jpg", "rb") if os.path.exists("signal_bg.jpg") else None,
        caption=signal,
        parse_mode='Markdown'
    )
    
    # Store the signal with advanced indicators
    if action_for_db and action_for_db != "HOLD":
        store_signal(user_id, pair, tf, action_for_db, current_price,
                     indicators["RSI"], indicators["EMA"], indicators["MA"],
                     indicators["Resistance"], indicators["Support"],
                     indicators["MACD"], indicators["MACD_Signal"],
                     indicators["Stoch_%K"], indicators["Stoch_%D"],
                     indicators["ADX"],
                     indicators["Bollinger_Upper"], indicators["Bollinger_Lower"])

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

# ... [rest of the bot setup code] ...