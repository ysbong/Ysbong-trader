# YSBONG TRADER™ BY PROSPERITY ENGINES™

import os, json, logging, asyncio, requests
from flask import Flask
from threading import Thread
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)

# === Flask Ping for Render Uptime ===
web_app = Flask(__name__)

@web_app.route('/')
def home():
    return "🤖 YSBONG TRADER™ is awake and running!"

def run_web():
    web_app.run(host="0.0.0.0", port=8080)

Thread(target=run_web).start()

# === Logging ===
logging.basicConfig(level=logging.INFO)

# === In-memory session ===
user_data = {}
usage_count = {}

# === User Key Storage ===
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
PAIRS = [
    "USD/JPY", "EUR/USD", "GBP/USD", "CAD/JPY", "USD/CAD",
    "AUD/CAD", "GBP/AUD", "EUR/AUD", "GBP/CAD", "CHF/JPY"
]
TIMEFRAMES = ["1MIN", "5MIN", "15MIN"]

# === Indicators ===
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

# === Candle Data ===
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

# === Start Command ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data[user_id] = {}
    usage_count[user_id] = usage_count.get(user_id, 0)

    if str(user_id) in saved_keys:
        user_data[user_id]["api_key"] = saved_keys[str(user_id)]
        user_data[user_id]["step"] = None
        kb = []
        for i in range(0, len(PAIRS), 2):
            row = [InlineKeyboardButton(PAIRS[i], callback_data=f"pair|{PAIRS[i]}")]
            if i + 1 < len(PAIRS):
                row.append(InlineKeyboardButton(PAIRS[i+1], callback_data=f"pair|{PAIRS[i+1]}"))
            kb.append(row)
        await update.message.reply_text("🔑 Welcome back! API key loaded.\n\n💱 Choose Currency Pair:",
                                        reply_markup=InlineKeyboardMarkup(kb))
        return

    kb = [[InlineKeyboardButton("✅ I Understand", callback_data="agree_disclaimer")]]
    await update.message.reply_text(
        "⚠️ DISCLAIMER — DEVELOPED BY PROSPERITY ENGINES™\n\n"
        "This bot provides educational signals only.\n"
        "Success comes from discipline and preparation.\n"
        "You are the engine of your prosperity. 💹",
        reply_markup=InlineKeyboardMarkup(kb)
    )

# === Button Handler ===
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
        tf_buttons = [[InlineKeyboardButton(tf, callback_data=f"timeframe|{tf}")] for tf in TIMEFRAMES]
        await context.bot.send_message(query.message.chat_id, "⏰ Choose Timeframe:", reply_markup=InlineKeyboardMarkup(tf_buttons))

    elif data.startswith("timeframe|"):
        user_data[user_id]["timeframe"] = data.split("|")[1]
        await context.bot.send_message(query.message.chat_id,
            "✅ Ready to generate signal!",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📲 GET SIGNAL", callback_data="get_signal")]])
        )

    elif data == "get_signal":
        await generate_signal(update, context)

# === API Key Input ===
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()
    if user_data.get(user_id, {}).get("step") == "awaiting_api":
        user_data[user_id]["api_key"] = text
        user_data[user_id]["step"] = None
        saved_keys[str(user_id)] = text
        save_keys(saved_keys)

        kb = []
        for i in range(0, len(PAIRS), 2):
            row = [InlineKeyboardButton(PAIRS[i], callback_data=f"pair|{PAIRS[i]}")]
            if i + 1 < len(PAIRS):
                row.append(InlineKeyboardButton(PAIRS[i+1], callback_data=f"pair|{PAIRS[i+1]}"))
            kb.append(row)

        await update.message.reply_text("🔐 API Key saved.\n💱 Choose Currency Pair:", reply_markup=InlineKeyboardMarkup(kb))

# === Signal Generator ===
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
        await context.bot.send_message(chat_id=chat_id,
            text="❌ API limit reached or invalid key. Please re-enter.")
        return

    indicators = calculate_indicators(result)
    current_price = float(result[0]["close"])
    action = "BUY 🔼" if current_price > indicators["EMA"] and indicators["RSI"] > 50 else "SELL 🔽"

    loading_msg = await context.bot.send_message(chat_id=chat_id, text="⏳ Generating signal in 3 seconds...")
    await asyncio.sleep(3)
    await loading_msg.delete()

    signal = (
        "📡 [YSBONG TRADER™ SIGNAL]\n\n"
        f"📍 PAIR:           {pair}\n"
        f"⏱️ TIMEFRAME:      {tf}\n"
        f"📊 ACTION:         {action}\n\n"
        f"— TECHNICALS —\n"
        f"🟩 MA: {indicators['MA']} | EMA: {indicators['EMA']}\n"
        f"📈 RSI: {indicators['RSI']}\n"
        f"🔺 Resistance: {indicators['Resistance']}\n"
        f"🔻 Support:    {indicators['Support']}"
    )
    await context.bot.send_message(chat_id=chat_id, text=signal)

    if usage_count[user_id] % 3 == 1:
        await context.bot.send_message(chat_id=chat_id,
            text="💡 Stay focused. Your consistency builds your legacy.\nBY: PROSPERITY ENGINES™")

# === Optional Reset Command ===
async def reset_api(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if user_id in saved_keys:
        saved_keys.pop(user_id)
        save_keys(saved_keys)
        await update.message.reply_text("🗑️ API key removed. Use /start to re-enter.")
    else:
        await update.message.reply_text("ℹ️ No API key saved.")

# === MAIN ===
if __name__ == '__main__':
    TOKEN = os.getenv("BOT_TOKEN")  # Secure for Render, or paste token directly for testing
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("resetapikey", reset_api))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_buttons))
    print("✅ YSBONG TRADER™ is LIVE...")
    app.run_polling()