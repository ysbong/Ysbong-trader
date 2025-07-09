# YSBONG TRADER™ WITH LEARNING MEMORY - BY PROSPERITY ENGINES™

import os, json, logging, asyncio, requests, sqlite3
from flask import Flask
from threading import Thread
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)

# === Flask ping for Render Uptime ===
web_app = Flask(__name__)

@web_app.route('/')
def home():
    return "🤖 YSBONG TRADER™ is awake and learning!"

def run_web():
    web_app.run(host="0.0.0.0", port=8080)

Thread(target=run_web).start()

# === SQLite Learning Memory ===
DB_FILE = "ysbong_memory.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            pair TEXT,
            timeframe TEXT,
            action TEXT,
            price REAL,
            rsi REAL,
            ema REAL,
            ma REAL,
            resistance REAL,
            support REAL,
            feedback TEXT DEFAULT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def store_signal(user_id, pair, tf, action, price, rsi, ema, ma, resistance, support):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO signals (user_id, pair, timeframe, action, price, rsi, ema, ma, resistance, support)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, pair, tf, action, price, rsi, ema, ma, resistance, support))
    conn.commit()
    conn.close()

def add_feedback(user_id, feedback):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        UPDATE signals
        SET feedback = ?
        WHERE user_id = ? AND feedback IS NULL
        ORDER BY id DESC
        LIMIT 1
    ''', (feedback, user_id))
    conn.commit()
    conn.close()

# === Init DB on startup ===
init_db()

# === Logging ===
logging.basicConfig(level=logging.INFO)

user_data = {}
usage_count = {}

# === API Key Storage ===
STORAGE_FILE = "user_keys.json"

def load_saved_keys():
    if os.path.exists(STORAGE_FILE):
        with open(STORAGE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_keys(data):
    with open(STORAGE_FILE, "w") as f:
        json.dump(data, f, indent=4)

saved_keys = load_saved_keys()

# === Constants ===
PAIRS = ["USD/JPY", "EUR/USD", "GBP/USD", "CAD/JPY", "USD/CAD",
         "AUD/CAD", "GBP/AUD", "EUR/AUD", "GBP/CAD", "CHF/JPY"]
TIMEFRAMES = ["1MIN", "5MIN", "15MIN"]

def calculate_ema(closes, period=9):
    ema = closes[0]
    k = 2 / (period + 1)
    for price in closes[1:]:
        ema = price * k + ema * (1 - k)
    return ema

def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50
    gains, losses = [], []
    for i in range(1, period + 1):
        delta = closes[-i] - closes[-i - 1]
        (gains if delta >= 0 else losses).append(abs(delta))
    avg_gain = sum(gains) / period if gains else 0.01
    avg_loss = sum(losses) / period if losses else 0.01
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_indicators(candles):
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
    params = {
        "symbol": symbol,
        "interval": "1min",
        "apikey": api_key,
        "outputsize": 30
    }
    try:
        res = requests.get(url, params=params)
        data = res.json()
        if "status" in data and data["status"] == "error":
            return "error", data.get("message", "API Error")
        return "ok", data.get("values", [])
    except:
        return "error", "Connection Error"

# === Telegram Handlers ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data[user_id] = {}
    usage_count[user_id] = usage_count.get(user_id, 0)
    if str(user_id) in saved_keys:
        user_data[user_id]["api_key"] = saved_keys[str(user_id)]
        user_data[user_id]["step"] = None
        kb = [[InlineKeyboardButton(PAIRS[i], callback_data=f"pair|{PAIRS[i]}"),
               InlineKeyboardButton(PAIRS[i+1], callback_data=f"pair|{PAIRS[i+1]}")]
              for i in range(0, len(PAIRS), 2)]
        await update.message.reply_text("🔑 API key loaded.\n💱 Choose Pair:", reply_markup=InlineKeyboardMarkup(kb))
        return
    kb = [[InlineKeyboardButton("✅ I Understand", callback_data="agree_disclaimer")]]
    await update.message.reply_text(
        "⚠️ DISCLAIMER\nThis bot provides educational signals only.\nYou are the engine of your prosperity.",
        reply_markup=InlineKeyboardMarkup(kb)
    )

async def handle_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()
    await query.message.delete()
    data = query.data
    if data == "agree_disclaimer":
        await context.bot.send_message(query.message.chat_id, "🔐 Please enter your API key:")
        user_data[user_id]["step"] = "awaiting_api"
    elif data.startswith("pair|"):
        user_data[user_id]["pair"] = data.split("|")[1]
        kb = [[InlineKeyboardButton(tf, callback_data=f"timeframe|{tf}")] for tf in TIMEFRAMES]
        await context.bot.send_message(query.message.chat_id, "⏰ Choose Timeframe:", reply_markup=InlineKeyboardMarkup(kb))
    elif data.startswith("timeframe|"):
        user_data[user_id]["timeframe"] = data.split("|")[1]
        await context.bot.send_message(query.message.chat_id,
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
        saved_keys[str(user_id)] = text
        save_keys(saved_keys)
        kb = [[InlineKeyboardButton(PAIRS[i], callback_data=f"pair|{PAIRS[i]}"),
               InlineKeyboardButton(PAIRS[i+1], callback_data=f"pair|{PAIRS[i+1]}")]
              for i in range(0, len(PAIRS), 2)]
        await update.message.reply_text("🔐 API Key saved.\n💱 Choose Currency Pair:", reply_markup=InlineKeyboardMarkup(kb))

async def generate_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    chat_id = query.message.chat_id
    usage_count[user_id] = usage_count.get(user_id, 0) + 1
    data = user_data.get(user_id, {})
    pair = data.get("pair", "EUR/USD")
    tf = data.get("timeframe", "1MIN")
    api_key = data.get("api_key")
    status, result = fetch_data(api_key, pair)
    if status == "error":
        user_data[user_id].pop("api_key", None)
        user_data[user_id]["step"] = "awaiting_api"
        await context.bot.send_message(chat_id=chat_id, text="❌ API limit reached. Please re-enter.")
        return
    indicators = calculate_indicators(result)
    current_price = float(result[0]["close"])
    action = "BUY 🔼" if current_price > indicators["EMA"] and indicators["RSI"] > 50 else "SELL 🔽"
    loading_msg = await context.bot.send_message(chat_id=chat_id, text="⏳ Generating signal in 3 seconds...")
    await asyncio.sleep(3)
    await loading_msg.delete()
    signal = (
        "📡 [YSBONG TRADER™ SIGNAL]\n\n"
        f"📍 PAIR:                     {pair}\n"
        f"⏱️ TIMEFRAME:       {tf}\n"
        f"📊 ACTION:               {action}\n\n"
        f"— TECHNICALS —\n"
        f"🟩 MA: {indicators['MA']} | EMA: {indicators['EMA']}\n"
        f"📈 RSI: {indicators['RSI']}\n"
        f"🔺 Resistance: {indicators['Resistance']}\n"
        f"🔻 Support:    {indicators['Support']}"
    )
    await context.bot.send_message(chat_id=chat_id, text=signal)

    store_signal(user_id, pair, tf, action, current_price,
                 indicators["RSI"], indicators["EMA"], indicators["MA"],
                 indicators["Resistance"], indicators["Support"])

    if usage_count[user_id] % 3 == 1:
        await context.bot.send_message(chat_id=chat_id,
            text="💡 Stay focused. Consistency builds your legacy.\nBY: PROSPERITY ENGINES™")

async def reset_api(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if user_id in saved_keys:
        saved_keys.pop(user_id)
        save_keys(saved_keys)
        await update.message.reply_text("🗑️ API key removed.")
    else:
        await update.message.reply_text("ℹ️ No API key found.")

async def feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args = context.args
    if not args or args[0] not in ["win", "loss"]:
        await update.message.reply_text("❗ Usage: /feedback win OR /feedback loss")
        return
    add_feedback(user_id, args[0])
    await update.message.reply_text(f"✅ Feedback saved: {args[0].upper()}")

# === Start Bot ===
if __name__ == '__main__':
    TOKEN = os.getenv("7618774950:AAF-SbIBviw3PPwQEGAFX_vsQZlgBVNNScI")  # Set this in Render Environment or paste your token
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("resetapikey", reset_api))
    app.add_handler(CommandHandler("feedback", feedback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_buttons))
    print("✅ YSBONG TRADER™ with learning is LIVE...")
    app.run_polling()