# main.py (Raw Code)
import os, json, logging, requests, sqlite3, datetime
import http.server, socketserver
from threading import Thread
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)

# === Lightweight Ping Server ===
class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write("🤖 YSBONG TRADER is awake.".encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run_ping_server():
    with socketserver.TCPServer(("", 9090), Handler) as httpd:
        httpd.serve_forever()

Thread(target=run_ping_server).start()

# === SQLite DB ===
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

user_data = {}
usage_count = {}
broadcasted_today = False
STORAGE_FILE = "user_keys.json"

def load_saved_keys():
    if os.path.exists(STORAGE_FILE):
        try:
            with open(STORAGE_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_keys(data):
    with open(STORAGE_FILE, "w") as f:
        json.dump(data, f, indent=4)

saved_keys = load_saved_keys()

PAIRS = ["USD/JPY", "EUR/USD", "GBP/USD", "CAD/JPY", "USD/CAD",
         "AUD/CAD", "GBP/AUD", "EUR/AUD", "GBP/CAD", "CHF/JPY"]
TIMEFRAMES = ["1MIN", "5MIN", "15MIN"]

INTERVAL_MAP = {
    "1MIN": "1min",
    "5MIN": "5min",
    "15MIN": "15min"
}

INTRO_MESSAGE = (
    "Hey guys\\! 👋\n\n"
    "I’ve been using this new signal bot on Telegram — it’s called *YSBONG TRADER™* 🤖\n\n"
    "✅ Real\\-time signals based on _live candle data_\n"
    "✅ Powered by AI with EMA, RSI, and MA\n"
    "✅ Connect your TwelveData API — FREE, no app required\n\n"
    "📲 Try it here: [Click Me](https://t.me/Bullish_bot)\n\n"
    "---\n\n"
    "🧠 *Tips for Beginners*:\n"
    "Practice first, deposit later\\. Start small\\.\n"
    "[Register here](https://pocket-friends.com/r/w2enb3tukw)\n\n"
    "Trade smart\\. Be patient\\. This bot is your assistant — not a crystal ball\\.\n\n"
    "— *YSBONG TRADER™* \\| powered by PROSPERITY ENGINES™"
)

# === Indicator Functions ===
def calculate_ema(closes, period=9):
    if not closes or len(closes) < period:
        return sum(closes) / len(closes) if closes else 0.0
    ema = sum(closes[:period]) / period
    k = 2 / (period + 1)
    for price in closes[period:]:
        ema = price * k + ema * (1 - k)
    return ema

def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [abs(d) if d < 0 else 0 for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    if avg_gain == 0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_indicators(candles):
    sorted_candles = list(reversed(candles))
    closes = [float(c['close']) for c in sorted_candles]
    highs = [float(c['high']) for c in candles]
    lows = [float(c['low']) for c in candles]
    return {
        "MA": round(sum(closes[-14:]) / min(len(closes), 14), 4),
        "EMA": round(calculate_ema(closes), 4),
        "RSI": round(calculate_rsi(closes), 2),
        "Resistance": round(max(highs) if highs else 0.0, 4),
        "Support": round(min(lows) if lows else 0.0, 4)
    }

def fetch_data(api_key, symbol, interval):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": INTERVAL_MAP.get(interval, "1min"),
        "apikey": api_key,
        "outputsize": 60
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        if "status" in data and data["status"] == "error":
            return "error", data.get("message", "API Error")
        return "ok", data.get("values", [])
    except Exception as e:
        return "error", str(e)

# === Telegram Handlers (simplified to core functions only) ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... same as earlier block ...
    pass

async def intro(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(INTRO_MESSAGE, parse_mode="MarkdownV2")

# === Main App Runner (Webhook version) ===
if __name__ == '__main__':
    TOKEN = "7618774950:AAF-SbIBviw3PPwQEGAFX_vsQZlgBVNNScI"
    app: Application = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("intro", intro))

    # More handlers go here...

    app.run_webhook(
        listen="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        webhook_url="https://ysbong-trader.onrender.com"
    )