# YSBONG TRADER™ – POWERED BY PROSPERITY ENGINES™

import os, json, logging, requests, sqlite3, datetime
import http.server, socketserver
from threading import Thread
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)

# === Lightweight Ping Server (for Render Uptime) ===
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

user_data = {}
usage_count = {} # Consider persisting this if you implement usage limits
broadcasted_today = False
STORAGE_FILE = "user_keys.json"

def load_saved_keys():
    if os.path.exists(STORAGE_FILE):
        try:
            with open(STORAGE_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {STORAGE_FILE}. Starting with empty keys.")
            return {}
    return {}

def save_keys(data):
    with open(STORAGE_FILE, "w") as f:
        json.dump(data, f, indent=4)

saved_keys = load_saved_keys()

PAIRS = ["USD/JPY", "EUR/USD", "GBP/USD", "CAD/JPY", "USD/CAD",
         "AUD/CAD", "GBP/AUD", "EUR/AUD", "GBP/CAD", "CHF/JPY"]
TIMEFRAMES = ["1MIN", "5MIN", "15MIN"]

# Mapping for TwelveData intervals
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
    "📲 Try it here: [Click Me](https://t.me/Bullish_bot)\n\n" # Replace with your bot's actual link
    "---\n\n"
    "🧠 *Tips for Beginners*:\n"
    "Practice first, deposit later\\. Start small\\.\n"
    "[Register here](https://pocket-friends.com/r/w2enb3tukw)\n\n" # Replace with your referral link
    "Trade smart\\. Be patient\\. This bot is your assistant — not a crystal ball\\.\n\n"
    "— *YSBONG TRADER™* \\| powered by PROSPERITY ENGINES™"
)

# --- Indicator Calculations ---
def calculate_ema(closes, period=9):
    if not closes:
        return 0.0
    if len(closes) < period:
        # Not enough data for EMA, return simple average or a default
        return sum(closes) / len(closes) if closes else 0.0

    # Calculate initial SMA for the first 'period' values
    ema = sum(closes[:period]) / period
    k = 2 / (period + 1)
    for price in closes[period:]:
        ema = price * k + ema * (1 - k)
    return ema

def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0 # Default RSI if not enough data

    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [abs(d) if d < 0 else 0 for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Calculate subsequent averages
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0 # Avoid division by zero, strong upward trend
    if avg_gain == 0:
        return 0.0 # Avoid division by zero, strong downward trend

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_indicators(candles):
    # Ensure candles are sorted from oldest to newest for correct indicator calculation
    # TwelveData returns newest first, so we reverse it for calculations
    sorted_candles = list(reversed(candles))

    closes = [float(c['close']) for c in sorted_candles]
    highs = [float(c['high']) for c in candles] # highs and lows can be from original order for support/resistance
    lows = [float(c['low']) for c in candles]

    if not closes:
        return {
            "MA": 0.0, "EMA": 0.0, "RSI": 50.0,
            "Resistance": 0.0, "Support": 0.0
        }

    return {
        "MA": round(sum(closes[-14:]) / min(len(closes), 14), 4), # Using 14-period MA
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
        "outputsize": 60 # Increased outputsize for better indicator calculation (e.g., 14-period RSI/MA, 9-period EMA)
    }
    try:
        res = requests.get(url, params=params, timeout=10) # Added timeout
        res.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = res.json()
        if "status" in data and data["status"] == "error":
            logging.error(f"TwelveData API error for {symbol} ({interval}): {data.get('message', 'Unknown API Error')}")
            return "error", data.get("message", "API Error")
        return "ok", data.get("values", [])
    except requests.exceptions.Timeout:
        logging.error(f"Timeout connecting to TwelveData for {symbol} ({interval}).")
        return "error", "Connection timed out. Please try again."
    except requests.exceptions.RequestException as e:
        logging.error(f"Network or API error fetching data for {symbol} ({interval}): {e}")
        return "error", f"Connection or API Error: {e}"
    except json.JSONDecodeError:
        logging.error(f"JSON decoding error from TwelveData response for {symbol} ({interval}): {res.text if 'res' in locals() else 'No response'}")
        return "error", "Invalid API response from TwelveData."
    except Exception as e:
        logging.error(f"An unexpected error occurred in fetch_data for {symbol} ({interval}): {e}")
        return "error", "An unexpected error occurred."

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
    await update.message.reply_text(INTRO_MESSAGE, parse_mode="MarkdownV2")

async def broadcast_intro(context: ContextTypes.DEFAULT_TYPE):
    global broadcasted_today
    if not broadcasted_today:
        logging.info("Starting daily intro broadcast.")
        for user_id_str in saved_keys: # Iterate through string keys
            try:
                user_id = int(user_id_str) # Convert back to int for send_message
                await context.bot.send_message(chat_id=user_id, text=INTRO_MESSAGE, parse_mode='MarkdownV2')
                logging.info(f"Sent intro to user {user_id}")
            except Exception as e:
                logging.error(f"❌ Failed to send intro to {user_id_str}: {e}")
        broadcasted_today = True
        logging.info("Daily intro broadcast completed.")

async def reset_intro_flag(context: ContextTypes.DEFAULT_TYPE):
    global broadcasted_today
    if broadcasted_today:
        broadcasted_today = False
        logging.info("Broadcasted today flag reset.")

async def disclaimer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    disclaimer_msg = (
        "⚠️ *Financial Risk Disclaimer*\n\n"
        "Trading involves real risk\\. This bot provides educational signals only\\.\n"
        "*Not financial advice\\.*\n\n"
        "📊 Be wise\\. Only trade what you can afford to lose\\.\n"
        "💡 Results depend on your discipline, not predictions\\."
    )
    await update.message.reply_text(disclaimer_msg, parse_mode='MarkdownV2')

async def handle_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()
    await query.message.delete() # Remove the old message/keyboard
    data = query.data

    if data == "agree_disclaimer":
        await context.bot.send_message(chat_id=query.message.chat_id, text="🔐 Please enter your TwelveData API key:")
        user_data[user_id]["step"] = "awaiting_api"
    elif data.startswith("pair|"):
        user_data[user_id]["pair"] = data.split("|")[1]
        kb = [[InlineKeyboardButton(tf, callback_data=f"timeframe|{tf}")] for tf in TIMEFRAMES]
        await context.bot.send_message(chat_id=query.message.chat_id, text="⏰ Choose Timeframe:", reply_markup=InlineKeyboardMarkup(kb))
    elif data.startswith("timeframe|"):
        user_data[user_id]["timeframe"] = data.split("|")[1]
        await context.bot.send_message(chat_id=query.message.chat_id,
            text="✅ Ready to generate signal!",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📲 GET SIGNAL", callback_data="get_signal")]])
        )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()

    if user_data.get(user_id, {}).get("step") == "awaiting_api":
        # Basic API key validation (you could add more robust validation)
        if len(text) == 32 and all(c.isalnum() for c in text): # TwelveData keys are 32 alphanumeric chars
            user_data[user_id]["api_key"] = text
            user_data[user_id]["step"] = None
            saved_keys[str(user_id)] = text # Save key as string
            save_keys(saved_keys)
            kb = [[InlineKeyboardButton(PAIRS[i], callback_data=f"pair|{PAIRS[i]}"),
                   InlineKeyboardButton(PAIRS[i+1], callback_data=f"pair|{PAIRS[i+1]}")]
                  for i in range(0, len(PAIRS), 2)]
            await update.message.reply_text("🔐 API Key saved.\n💱 Choose Currency Pair:", reply_markup=InlineKeyboardMarkup(kb))
            await update.message.reply_text(INTRO_MESSAGE, parse_mode='MarkdownV2')
        else:
            await update.message.reply_text("That doesn't look like a valid API key. Please try again.")
    else:
        # Default response for unhandled text messages
        await update.message.reply_text("I didn't understand that. Please use the buttons or commands like /start.")

async def get_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()
    await query.message.delete()

    user_info = user_data.get(user_id)
    if not user_info or "api_key" not in user_info or "pair" not in user_info or "timeframe" not in user_info:
        await context.bot.send_message(chat_id=user_id, text="ℹ️ It seems we lost track of your selection. Please start again with /start to set your preferences.")
        return

    api_key = user_info["api_key"]
    pair = user_info["pair"]
    timeframe = user_info["timeframe"]

    await context.bot.send_message(chat_id=user_id, text=f"📊 Fetching data for *{pair}* \\({timeframe}\\)... Please wait\\.", parse_mode='MarkdownV2')

    # TwelveData expects symbols without slashes for most forex pairs (e.g., EURUSD instead of EUR/USD)
    twelvedata_symbol = pair.replace("/", "")
    status, candles = fetch_data(api_key, twelvedata_symbol, timeframe)

    if status == "error":
        await context.bot.send_message(chat_id=user_id, text=f"❌ Error fetching data: {candles}. Please check your API key or try again later.")
        logging.error(f"Error for user {user_id}: {candles}")
        # Offer to retry or restart if an error occurs
        kb_error = [[InlineKeyboardButton("🔄 Try Again", callback_data="get_signal")],
                    [InlineKeyboardButton("↩️ Start Over", callback_data="start_new_signal")]]
        await context.bot.send_message(chat_id=user_id, text="Would you like to try fetching the signal again or start a new selection?", reply_markup=InlineKeyboardMarkup(kb_error))
        return

    if not candles:
        await context.bot.send_message(chat_id=user_id, text=f"😔 No sufficient data available for *{pair}* \\({timeframe}\\) to generate a signal\\.", parse_mode='MarkdownV2')
        # Offer to retry or restart if no data
        kb_no_data = [[InlineKeyboardButton("↩️ Start Over", callback_data="start_new_signal")]]
        await context.bot.send_message(chat_id=user_id, text="Please select another pair or timeframe.", reply_markup=InlineKeyboardMarkup(kb_no_data))
        return

    try:
        indicators = calculate_indicators(candles)
        # The last candle in the `candles` list from TwelveData is the most recent.
        current_price = float(candles[0]['close']) if candles else 0.0

        # === Implement your trading logic here ===
        # This is a basic example; refine based on your desired strategy.
        action = "HOLD"
        reasons = []

        # Example logic (you need to define your own)
        if indicators["RSI"] < 30 and current_price < indicators["Support"]:
            action = "BUY"
            reasons.append("RSI is oversold and price is near support.")
        elif indicators["RSI"] > 70 and current_price > indicators["Resistance"]:
            action = "SELL"
            reasons.append("RSI is overbought and price is near resistance.")
        elif indicators["EMA"] > current_price and indicators["MA"] > current_price:
            action = "SELL"
            reasons.append("EMA and MA are above current price (bearish trend).")
        elif indicators["EMA"] < current_price and indicators["MA"] < current_price:
            action = "BUY"
            reasons.append("EMA and MA are below current price (bullish trend).")
        elif indicators["EMA"] < indicators["MA"] and current_price < indicators["MA"]:
            action = "SELL"
            reasons.append("EMA crossed below MA, and price is below MA.")
        elif indicators["EMA"] > indicators["MA"] and current_price > indicators["MA"]:
            action = "BUY"
            reasons.append("EMA crossed above MA, and price is above MA.")

        if not reasons:
            reasons.append("Based on current indicator analysis.")

        signal_message = (
            f"📈 *Signal for {pair}* \\({timeframe}\\)\n"
            f"---\\n"
            f"💰 *Current Price*: `{current_price:.4f}`\n"
            f"---\\n"
            f"🤖 *Indicators*:\n"
            f"  • RSI: `{indicators['RSI']:.2f}`\n"
            f"  • EMA: `{indicators['EMA']:.4f}`\n"
            f"  • MA: `{indicators['MA']:.4f}`\n"
            f"  • Resistance: `{indicators['Resistance']:.4f}`\n"
            f"  • Support: `{indicators['Support']:.4f}`\n"
            f"---\\n"
            f"💡 *Action*: *{action}*\n"
            f"💬 *Reason\\(s\\)*: {', '.join(reasons)}\n"
            f"---\\n"
            f"✅ Trade smart\\. Be patient\\. This bot is your assistant — not a crystal ball\\."
        )

        # Save the signal to the database
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''
            INSERT INTO signals (user_id, pair, timeframe, action, price, rsi, ema, ma, resistance, support)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, pair, timeframe, action, current_price, indicators['RSI'], indicators['EMA'],
              indicators['MA'], indicators['Resistance'], indicators['Support']))
        conn.commit()
        conn.close()
        logging.info(f"Signal saved for user {user_id}: {pair} {timeframe} {action}")

        await context.bot.send_message(chat_id=user_id, text=signal_message, parse_mode='MarkdownV2')

    except Exception as e:
        logging.error(f"Error processing signal for user {user_id} - Pair: {pair}, Timeframe: {timeframe}: {e}", exc_info=True)
        await context.bot.send_message(chat_id=user_id, text="❌ An error occurred while generating the signal. Please try again.")

    # Offer to generate another signal or go back to main menu
    kb_restart = [[InlineKeyboardButton("🔄 Get another signal", callback_data="start_new_signal")]]
    await context.bot.send_message(chat_id=user_id, text="What next?", reply_markup=InlineKeyboardMarkup(kb_restart))

async def start_new_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()
    await query.message.delete()
    kb = [[InlineKeyboardButton(PAIRS[i], callback_data=f"pair|{PAIRS[i]}"),
           InlineKeyboardButton(PAIRS[i+1], callback_data=f"pair|{PAIRS[i+1]}")]
          for i in range(0, len(PAIRS), 2)]
    await context.bot.send_message(chat_id=user_id, text="💱 Choose Pair:", reply_markup=InlineKeyboardMarkup(kb))

# === Start Bot via Webhook ===

if __name__ == '__main__':
    # It's highly recommended to use environment variables for your token
    # For example: TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TOKEN = "7618774950:AAF-SbIBviw3PPwQEGAFX_vsQZlgBVNNScI" # Replace with your actual bot token

    if not TOKEN:
        logging.critical("TELEGRAM_BOT_TOKEN environment variable not set. Exiting.")
        exit(1)

    app: Application = ApplicationBuilder().token(TOKEN).build()

    # Command Handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("intro", intro))
    app.add_handler(CommandHandler("disclaimer", disclaimer))

    # Message Handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Callback Query Handlers
    app.add_handler(CallbackQueryHandler(handle_buttons, pattern="^(pair|timeframe|agree_disclaimer).*"))
    app.add_handler(CallbackQueryHandler(get_signal, pattern="^get_signal$"))
    app.add_handler(CallbackQueryHandler(start_new_signal, pattern="^start_new_signal$"))

    # Job Queue for scheduled tasks
    # Broadcast intro at 10:00 AM daily (local time of server)
    app.job_queue.run_daily(broadcast_intro, time=datetime.time(10, 0), name="daily_intro_broadcast")
    # Reset broadcast flag at 00:05 AM daily
    app.job_queue.run_daily(reset_intro_flag, time=datetime.time(0, 5), name="reset_broadcast_flag")

    print("✅ YSBONG TRADER™ Webhook is LIVE...")
    logging.info("YSBONG TRADER™ Webhook is LIVE.")

    # Remember to set your webhook_url to your Render deployment URL
    # e.g., "https://your-render-app-name.onrender.com/{TOKEN}"
    app.run_webhook(
        listen="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)), # Use PORT env var for Render
        url_path=TOKEN,
        webhook_url=f"https://ysbong-trader.onrender.com/{TOKEN}" # Replace with your actual Render URL
    )

