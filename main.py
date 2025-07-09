# YSBONG TRADER™ – POWERED BY PROSPERITY ENGINES™

import os, json, logging, asyncio, requests, sqlite3, datetime
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

init_db()

# === Logging ===
logging.basicConfig(level=logging.INFO)

user_data = {}
usage_count = {}
broadcasted_today = False
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

PAIRS = ["USD/JPY", "EUR/USD", "GBP/USD", "CAD/JPY", "USD/CAD",
         "AUD/CAD", "GBP/AUD", "EUR/AUD", "GBP/CAD", "CHF/JPY"]
TIMEFRAMES = ["1MIN", "5MIN", "15MIN"]

INTRO_MESSAGE = """
Hey guys! 👋

I’ve been using this new signal bot on Telegram — it’s called **YSBONG TRADER™** 🤖

✅ Real-time signals based on *live candle data* (not simulation or OTC)  
✅ Powered by AI with indicators like EMA, RSI, and MA  
✅ Just connect your free TwelveData API key — no app to install, no cost to use  
✅ And yes, it’s 100% FREE. No subscriptions. No upsells. Not for sale.

The bot reads your live market data using your API key — so what you see reflects real-time chart movement.

Want to check it out?  
📲 https://t.me/Bullish_bot

Just send your API key and follow the steps. That’s it.

---

🧪 **New to trading?**

Start by learning. Practice first. Understand the charts.

👉 Create your account here:  
https://pocket-friends.com/r/w2enb3tukw

💵 You can deposit later — even just $10 — when you're ready.

---

⚠️ **Important Reminders:**

• Don’t overtrade — 3 to 4 sessions per day is enough  
• Stay patient. Stay disciplined. Respect the market.  
• Yesterday’s move is not today’s guarantee.

This bot gives analysis — not magic. Use it wisely, and with a clear mind. 🧠

---

You’re not racing anyone.  
You're building a future.  
And your calm decisions today... shape that future tomorrow.

– **YSBONG TRADER™** | powered by PROSPERITY ENGINES™
"""

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

async def intro(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(INTRO_MESSAGE, parse_mode="Markdown")

async def broadcast_intro(context: ContextTypes.DEFAULT_TYPE):
    global broadcasted_today
    if not broadcasted_today:
        for user_id in saved_keys:
            try:
                await context.bot.send_message(chat_id=user_id, text=INTRO_MESSAGE, parse_mode='Markdown')
            except Exception as e:
                print(f"❌ Failed to send intro to {user_id}: {e}")
        broadcasted_today = True

async def reset_intro_flag(context: ContextTypes.DEFAULT_TYPE):
    global broadcasted_today
    broadcasted_today = False

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
        await update.message.reply_text(INTRO_MESSAGE, parse_mode='Markdown')

# === Start Bot ===

if __name__ == '__main__':
    TOKEN = "7618774950:AAF-SbIBviw3PPwQEGAFX_vsQZlgBVNNScI"
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("intro", intro))
    app.add_handler(CommandHandler("disclaimer", disclaimer))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_buttons))

    # ✅ Schedule auto intro broadcast
    app.job_queue.run_daily(broadcast_intro, time=datetime.time(10, 0))
    app.job_queue.run_daily(reset_intro_flag, time=datetime.time(0, 5))

    print("✅ YSBONG TRADER™ is LIVE...")
    app.run_polling()