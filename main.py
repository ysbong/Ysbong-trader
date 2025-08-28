import os, json, logging, asyncio, requests, time
import numpy as np
from flask import Flask
from threading import Thread, Lock
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ChatMember
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)

# === Logging Setup (Critical to be first) ===
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("⚡ Initializing YSBONG TRADER™ - AI Edition")

# Data Handling Imports
import pandas as pd
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import random
import math
from functools import wraps

# Type hinting imports
from typing import List, Tuple, Union, Optional, Dict, Callable

import asyncio
import nest_asyncio
nest_asyncio.apply()

# === ML Model Imports ===
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# === PostgreSQL Database Imports ===
import psycopg2
from psycopg2 import sql
from psycopg2.pool import SimpleConnectionPool

# Connection pool for database
db_pool = None
db_lock = Lock()

# === Smart Signal Decorator ===
def smart_signal_strategy(func: Callable) -> Callable:
    """Decorator to enhance signal generation with advanced strategies"""
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        query = update.callback_query
        user_id = query.from_user.id
        chat_id = query.message.chat_id
        
        # Get user data from database
        user_info = get_user_info(user_id)
        if not user_info:
            await context.bot.send_message(chat_id, text="❌ User not found. Please start with /start.")
            return
            
        pair = user_info.get("current_pair", "EUR/USD")
        tf = user_info.get("current_timeframe", "1MIN")
        api_key = user_info.get("api_key")

        if not api_key:
            await context.bot.send_message(chat_id, text="❌ API key not found. Please set your API key using /start.")
            return

        # Convert timeframe to interval format
        def timeframe_to_interval(tf):
            mapping = {"1MIN": "1min", "5MIN": "5min", "15MIN": "15min", "30MIN": "30min", "45MIN": "45min", "1H":"1h"}
            return mapping.get(tf, "1min")

        # Show loading animation
        loading_frames = [
            "[░░░░░░░░░░] 0%",
            "[▓░░░░░░░░░] 10%",
            "[▓▓░░░░░░░░] 20%",
            "[▓▓▓░░░░░░░] 30%",
            "[▓▓▓▓░░░░░░] 40%",
            "[▓▓▓▓▓░░░░░] 50%",
            "[▓▓▓▓▓▓░░░░] 60%",
            "[▓▓▓▓▓▓▓░░░] 70%",
            "[▓▓▓▓▓▓▓▓░░] 80%",
            "[▓▓▓▓▓▓▓▓▓░] 90%",
            "[▓▓▓▓▓▓▓▓▓▓]💥100%"
        ]
        loading_msg = await context.bot.send_message(chat_id, text=f"🔍 Analyzing market... {loading_frames[0]}")
        
        # Animate loading bar
        for i in range(1, len(loading_frames)):
            await asyncio.sleep(0.1)
            try:
                await loading_msg.edit_text(text=f"🔍 Analyzing market... {loading_frames[i]}")
            except Exception as e:
                logger.error(f"Error editing loading message: {e}")
                break

        # Fetch data
        status, result = fetch_data(api_key, pair, interval=timeframe_to_interval(tf))
        if status == "error":
            try:
                await loading_msg.delete()
            except:
                pass
            await context.bot.send_message(chat_id, text=f"❌ {result}")
            return

        # Calculate indicators
        try:
            indicators = calculate_indicators(result)
            current_price = float(result[0]["close"])
            
            # === Ensemble Hybrid Model Prediction ===
            # Prepare features for ML model
            feature_order = ['MA', 'EMA', 'RSI', 'Resistance', 'Support', 
                            'VWAP', 'MACD_LINE', 'MACD_SIGNAL', 'MACD_HIST']
            features = np.array([indicators[key] for key in feature_order]).reshape(1, -1)
            
            # Predict using ensemble model
            action_for_db, confidence = predict_with_ensemble(features)
            
            # Map to display action
            if action_for_db == "BUY":
                action = "HIGHER/BUY 🟢"
            elif action_for_db == "SELL":
                action = "LOWER/SELL 🔴"
            
            confidence_level = f"{confidence:.0%}"

        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            # Fallback to rule-based logic if ML fails
            buy_signals = 0
            sell_signals = 0
            
            # Price vs EMA (Trend Direction)
            if current_price > indicators["EMA"]:
                buy_signals += 1.5
            else:
                sell_signals += 1.5
                
            # RSI Momentum
            if indicators["RSI"] > 55:
                buy_signals += 1.2
            elif indicators["RSI"] < 45:
                sell_signals += 1.2
                
            # MACD Crossover Detection
            if indicators["MACD_HIST"] > 0 and indicators["MACD_LINE"] > indicators["MACD_SIGNAL"]:
                buy_signals += 1.3
            elif indicators["MACD_HIST"] < 0 and indicators["MACD_LINE"] < indicators["MACD_SIGNAL"]:
                sell_signals += 1.3
                
            # Price vs VWAP (Market Sentiment)
            if current_price > indicators["VWAP"]:
                buy_signals += 0.8
            else:
                sell_signals += 0.8
                
            # Support/Resistance Levels
            support_distance = abs(current_price - indicators["Support"])
            resistance_distance = abs(current_price - indicators["Resistance"])
            
            if support_distance < resistance_distance * 0.7:
                buy_signals += 1.0
            elif resistance_distance < support_distance * 0.7:
                sell_signals += 1.0
                
            # Determine final signal
            confidence_val = abs(buy_signals - sell_signals) / max(buy_signals + sell_signals, 1)
            
            if buy_signals >= sell_signals:
                action = "HIGHER/BUY 🟢"
                action_for_db = "BUY"
                confidence_level = f"{min(95, int(confidence_val * 100))}%"
            else:
                action = "LOWER/SELL 🔴"
                action_for_db = "SELL"
                confidence_level = f"{min(95, int(confidence_val * 100))}%"
            

        # Format and send signal
        flagged_pair = get_flagged_pair_name(pair)

        signal = (
            "🚀 *YSBONG TRADER™ SMART SIGNAL*\n"
            f"━━━━━━━━━━━━━━━━━━━\n" 
            f"💹 PAIR: {flagged_pair}\n"
            f"⏱ TIMEFRAME: {tf}\n"
            f"🧨 ACTION: {action}\n"
            f"🎯 CONFIDENCE: {confidence_level}\n"
            f"━━━━━━━━━━━━━━━━━━━\n" 
            f"📊 *MARKET ANALYSIS*\n"
            f"💰 Price: {current_price:.4f}\n"
            f"📉 RSI: {indicators['RSI']:.1f} ({'Overbought' if indicators['RSI'] > 70 else 'Oversold' if indicators['RSI'] < 30 else 'Neutral'})\n"
            f"📈 EMA: {indicators['EMA']:.4f}\n"
            f"📊 VWAP: {indicators['VWAP']:.4f}\n"
            f"📈 MACD: {indicators['MACD_HIST']:.4f} ({'Bullish' if indicators['MACD_HIST'] > 0 else 'Bearish'})\n"
            f"🛡️ Support: {indicators['Support']:.4f}\n"
            f"🚧 Resistance: {indicators['Resistance']:.4f}\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"🤖 *AI Model Used:* Hybrid Ensemble (RFC+NN)\n"
            f"🔥Your (Win/Loss) clicks directly fuel the AI's learning engine...\n"
            f"☣️ Avoid overtrading! More trades don't mean more profits, they usually mean more mistakes...\n"
        )
        
        # Delete loading message before sending signal
        try:
            await loading_msg.delete()
        except Exception as e:
            logger.warning(f"Failed to delete loading message: {e}")
        
        # Prepare feedback buttons
        feedback_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🤑 Win", callback_data=f"feedback|win"),
             InlineKeyboardButton("😭 Loss", callback_data=f"feedback|loss")]
        ])
        await context.bot.send_message(chat_id=chat_id, text=signal, 
                                      reply_markup=feedback_keyboard, parse_mode='Markdown')
        
        # Store the signal
        store_signal(user_id, pair, tf, action_for_db, current_price, indicators)

    return wrapper

# === Ensemble Model Functions ===
def predict_with_ensemble(features: np.ndarray) -> Tuple[str, float]:
    """
    Predicts trading action using hybrid ensemble of Random Forest and Neural Network.
    Returns tuple: (action, confidence)
    """
    # Load models if not already loaded
    global ensemble_model
    
    if ensemble_model is None:
        load_ensemble_model()
        
    if ensemble_model is None:
        raise Exception("Ensemble model failed to load")
    
    # Predict probabilities
    probabilities = ensemble_model.predict_proba(features)[0]
    
    # Get class with highest probability
    class_index = np.argmax(probabilities)
    confidence = probabilities[class_index]
    
    # Map index to action
    actions = ['BUY', 'SELL']
    return actions[class_index], confidence

def load_ensemble_model() -> None:
    """Loads the ensemble model from file or creates a new one if not found"""
    global ensemble_model
    model_path = "ensemble_model.pkl"
    
    try:
        # Try to load existing model
        ensemble_model = joblib.load(model_path)
        logger.info("✅ Ensemble model loaded from file")
    except Exception as e:
        try:
            logger.warning(f"⚠️ Model file not found or corrupted: {e}. Training new ensemble model...")
            ensemble_model = train_new_model()
            joblib.dump(ensemble_model, model_path)
            logger.info("✅ New ensemble model trained and saved")
        except Exception as e:
            logger.error(f"❌ Error training new model: {e}")
            ensemble_model = None

def train_new_model() -> Pipeline:
    """Trains a new ensemble model using historical data"""
    logger.info("⚙️ Training new hybrid ensemble model...")
    
    # Fetch historical data from database
    historical_data = fetch_historical_data()
    
    if historical_data.empty:
        raise Exception("No historical data available for training")
    
    # Prepare features and labels
    feature_order = ['MA', 'EMA', 'RSI', 'Resistance', 'Support', 
                     'VWAP', 'MACD_LINE', 'MACD_SIGNAL', 'MACD_HIST']
    
    # Add additional features
    historical_data['price_vs_ema'] = historical_data['price'] - historical_data['EMA']
    historical_data['price_vs_vwap'] = historical_data['price'] - historical_data['VWAP']
    historical_data['macd_cross'] = np.where(historical_data['MACD_LINE'] > historical_data['MACD_SIGNAL'], 1, -1)
    
    feature_order += ['price_vs_ema', 'price_vs_vwap', 'macd_cross']
    
    X = historical_data[feature_order]
    y = historical_data['action_for_db']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create hybrid model pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ensemble', HybridEnsembleModel())
    ])
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"🧠 Model trained. Accuracy: {accuracy:.2%}")
    
    return model

def fetch_historical_data() -> pd.DataFrame:
    """Fetches historical trading data from database for model training"""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return pd.DataFrame()
            
        query = """
        SELECT 
            MA, EMA, RSI, Resistance, Support, VWAP, 
            macd_line, macd_signal, macd_hist,
            price,
            action_for_db
        FROM signals
        WHERE feedback IS NOT NULL
        """
        df = pd.read_sql_query(query, conn)
        
        # Filter only valid actions
        valid_actions = ['BUY', 'SELL']
        df = df[df['action_for_db'].isin(valid_actions)]
        
        logger.info(f"📊 Loaded {len(df)} historical records for training")
        return df
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

class HybridEnsembleModel:
    """Hybrid ensemble model combining Random Forest and Neural Network"""
    def __init__(self):
        self.rfc = RandomForestClassifier(
            n_estimators=200, 
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            max_features='sqrt'
        )
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42,
            early_stopping=True
        )
        
    def fit(self, X, y):
        # Train both models
        self.rfc.fit(X, y)
        self.mlp.fit(X, y)
        
    def predict_proba(self, X):
        # Get probabilities from both models
        rfc_proba = self.rfc.predict_proba(X)
        mlp_proba = self.mlp.predict_proba(X)
        
        # Weighted average probabilities
        return (0.6 * rfc_proba + 0.4 * mlp_proba)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

# Initialize ensemble model
ensemble_model = None

# === Channel Membership Requirement ===
CHANNEL_USERNAME = "@ProsperityEngines"  # Replace with your channel username
CHANNEL_LINK = "https://t.me/ProsperityEngines"  # Replace with your channel link

async def is_user_joined(user_id: int, bot) -> bool:
    """Checks if a user is a member of the required Telegram channel."""
    try:
        member = await bot.get_chat_member(chat_id=CHANNEL_USERNAME, user_id=user_id)
        return member.status in [ChatMember.MEMBER, ChatMember.OWNER, ChatMember.ADMINISTRATOR]
    except Exception as e:
        logging.error(f"Error checking membership for user {user_id}: {e}")
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
def home() -> str:
    """Root endpoint for the web application."""
    return "YSBONG TRADER™ (Active) is awake and scanning!"

@web_app.route("/health")
def health() -> Tuple[str, int]:
    """Health check endpoint."""
    return "✅ YSBONG™ is alive and kicking!", 200

def run_web() -> None:
    """Runs the Flask web application in a separate thread."""
    port = int(os.environ.get("PORT", 8080))
    web_app.run(host="0.0.0.0", port=port)

# Start the Flask app in a separate thread
Thread(target=run_web).start()

# === PostgreSQL Database Connection ===
def init_db_pool():
    """Initializes the PostgreSQL database connection pool"""
    global db_pool
    try:
        # Get database URL from environment (provided by Render)
        database_url = os.environ.get('DATABASE_URL')
        
        if not database_url:
            logger.error("DATABASE_URL environment variable not set")
            return None
            
        # Create connection pool
        db_pool = SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=database_url,
            row_factory=dict_row
        )
        logger.info("✅ PostgreSQL connection pool initialized")
        return db_pool
    except Exception as e:
        logger.error(f"Error initializing PostgreSQL connection pool: {e}")
        return None

def get_db_connection():
    """Gets a connection from the connection pool"""
    global db_pool
    if db_pool is None:
        init_db_pool()
    
    try:
        return db_pool.getconn()
    except Exception as e:
        logger.error(f"Error getting database connection: {e}")
        return None

def return_db_connection(conn):
    """Returns a connection to the pool"""
    global db_pool
    if db_pool and conn:
        db_pool.putconn(conn)

def init_db() -> None:
    """Initializes the PostgreSQL database tables using psycopg3."""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            logger.error("Could not establish database connection")
            return
            
        with conn.cursor() as cur:
            # Create signals table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER,
                    pair TEXT,
                    timeframe TEXT,
                    action_for_db TEXT,
                    price REAL,
                    rsi REAL,
                    ema REAL,
                    ma REAL,
                    resistance REAL,
                    support REAL,
                    vwap REAL,
                    macd_line REAL,
                    macd_signal REAL,
                    macd_hist REAL,
                    feedback TEXT DEFAULT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create user_api_keys table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS user_api_keys (
                    user_id INTEGER PRIMARY KEY,
                    api_key TEXT NOT NULL,
                    agreed_to_disclaimer BOOLEAN DEFAULT FALSE,
                    current_pair TEXT DEFAULT 'EUR/USD',
                    current_timeframe TEXT DEFAULT '1MIN',
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cur.execute('CREATE INDEX IF NOT EXISTS idx_signals_user_id ON signals(user_id)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_user_api_keys_last_active ON user_api_keys(last_active)')
            
            conn.commit()
            logger.info("✅ PostgreSQL database initialized successfully")
            
    except Exception as e:
        logger.error(f"PostgreSQL initialization error: {e}")
    finally:
        if conn:
            return_db_connection(conn)

# Initialize the database
init_db()

def get_user_info(user_id: int) -> Optional[Dict]:
    """Gets user information from the database"""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return None
            
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, api_key, agreed_to_disclaimer, current_pair, current_timeframe FROM user_api_keys WHERE user_id = %s",
                (user_id,)
            )
            result = cur.fetchone()
            return result
    except Exception as e:
        logger.error(f"Error getting user info: {e}")
        return None
    finally:
        if conn:
            return_db_connection(conn)

def save_user_info(user_id: int, api_key: str = None, agreed_to_disclaimer: bool = None, 
                  current_pair: str = None, current_timeframe: str = None) -> bool:
    """Saves user information to the database"""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return False
            
        with conn.cursor() as cur:
            # Build the update query dynamically based on provided parameters
            update_fields = []
            params = []
            
            if api_key is not None:
                update_fields.append("api_key = %s")
                params.append(api_key)
                
            if agreed_to_disclaimer is not None:
                update_fields.append("agreed_to_disclaimer = %s")
                params.append(agreed_to_disclaimer)
                
            if current_pair is not None:
                update_fields.append("current_pair = %s")
                params.append(current_pair)
                
            if current_timeframe is not None:
                update_fields.append("current_timeframe = %s")
                params.append(current_timeframe)
                
            # Always update last_active
            update_fields.append("last_active = CURRENT_TIMESTAMP")
            
            # Add user_id to params
            params.append(user_id)
            
            if update_fields:
                query = f"""
                    INSERT INTO user_api_keys (user_id, api_key, agreed_to_disclaimer, current_pair, current_timeframe, last_active)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id) 
                    DO UPDATE SET {', '.join(update_fields)}
                """
                
                # If it's an insert, we need all values
                if api_key is None:
                    user_info = get_user_info(user_id)
                    if user_info:
                        api_key = user_info.get('api_key')
                    else:
                        api_key = ''
                
                if agreed_to_disclaimer is None:
                    user_info = get_user_info(user_id)
                    if user_info:
                        agreed_to_disclaimer = user_info.get('agreed_to_disclaimer', False)
                    else:
                        agreed_to_disclaimer = False
                        
                if current_pair is None:
                    user_info = get_user_info(user_id)
                    if user_info:
                        current_pair = user_info.get('current_pair', 'EUR/USD')
                    else:
                        current_pair = 'EUR/USD'
                        
                if current_timeframe is None:
                    user_info = get_user_info(user_id)
                    if user_info:
                        current_timeframe = user_info.get('current_timeframe', '1MIN')
                    else:
                        current_timeframe = '1MIN'
                
                cur.execute(query, (user_id, api_key, agreed_to_disclaimer, current_pair, current_timeframe, *params))
            else:
                # Just update last_active
                cur.execute(
                    "UPDATE user_api_keys SET last_active = CURRENT_TIMESTAMP WHERE user_id = %s",
                    (user_id,)
                )
            
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Error saving user info: {e}")
        return False
    finally:
        if conn:
            return_db_connection(conn)

def remove_user_api_key(user_id: int) -> bool:
    """Removes a user's API key from the database"""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return False
            
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE user_api_keys SET api_key = NULL, agreed_to_disclaimer = FALSE WHERE user_id = %s",
                (user_id,)
            )
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Error removing user API key: {e}")
        return False
    finally:
        if conn:
            return_db_connection(conn)

def cleanup_inactive_users():
    """Cleans up users who haven't been active for more than 30 days"""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return
            
        with conn.cursor() as cur:
            # Delete users who haven't been active for 30 days
            cur.execute(
                "DELETE FROM user_api_keys WHERE last_active < CURRENT_TIMESTAMP - INTERVAL '30 days'"
            )
            deleted_count = cur.rowcount
            
            # Also clean up old signals
            cur.execute(
                "DELETE FROM signals WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '90 days'"
            )
            signals_count = cur.rowcount
            
            conn.commit()
            logger.info(f"🧹 Cleaned up {deleted_count} inactive users and {signals_count} old signals")
    except Exception as e:
        logger.error(f"Error cleaning up inactive users: {e}")
    finally:
        if conn:
            return_db_connection(conn)

# ===== FLAG MAPPING =====
# Corrected mapping to use currency codes as keys
CURRENCY_FLAGS = {
    "USD": "🇺🇸",
    "EUR": "🇪🇺",
    "GBP": "🇬🇧",
    "JPY": "🇯🇵",
    "CHF": "🇨🇭",
    "CAD": "🇨🇦",
    "AUD": "🇦🇺",
    "NZD": "🇳🇿",
    "SGD": "🇸🇬",
    "HKD": "🇭🇰"
}

# Mapping normal letters → tiny unicode letters
TINY_MAP = str.maketrans(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/",
    "ᴀʙᴄᴅᴇꜰɢʜɪᴊᴋʟᴍɴᴏᴘǫʀsᴛᴜᴠᴡxʏᴢ"
    "ᴀʙᴄᴅᴇꜰɢʜɪᴊᴋʟᴍɴᴏᴘǫʀsᴛᴜᴠᴡxʏᴢ"
    "₀₁₂₃₄₅₆₇₈₉∕"
)

def to_tiny(text: str) -> str:
    """Convert normal text to tiny unicode text"""
    return text.translate(TINY_MAP)

def get_flagged_pair_name(pair: str) -> str:
    """Return pair with flags + tiny text"""
    base, quote = pair.split("/")
    flag1 = CURRENCY_FLAGS.get(base, "")
    flag2 = CURRENCY_FLAGS.get(quote, "")
    return f"{flag1}{flag2}{to_tiny(pair)}"

# === Constants ===
PAIRS: List[str] = [
    "AUD/CAD", "AUD/CHF", "AUD/JPY", "AUD/NZD", "AUD/USD",
"CAD/CHF", "CAD/JPY",
"CHF/JPY",
"EUR/AUD", "EUR/CAD", "EUR/CHF", "EUR/GBP", "EUR/JPY", "EUR/NZD", "EUR/USD",
"GBP/AUD", "GBP/CAD", "GBP/JPY", "GBP/NZD", "GBP/SGD", "GBP/USD",
"NZD/CAD", "NZD/CHF", "NZD/JPY", "NZD/USD",
"USD/CAD", "USD/CHF", "USD/HKD", "USD/JPY", "USD/SGD"
]
TIMEFRAMES: List[str] = ["1MIN", "5MIN", "15MIN", "30MIN", "45MIN", "1H"]

# === TwelveData API Fetcher ===

def fetch_data(api_key: str, symbol: str, interval: str = "1min", outputsize: int = 100) -> Tuple[str, Union[List[dict], str]]:
    """
    Fetches candlestick data from TwelveData.
    Returns a tuple: ("success", data) or ("error", error_message)
    """
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=15) # Increased timeout
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if data.get("status") == "error":
            error_msg = data.get("message", "Unknown API error.")
            if "API key" in error_msg:
                return "error", "Invalid API key. Please check your key and try again."
            return "error", error_msg
        
        candles = data.get("values", [])
        if not candles:
            return "error", "No data returned for the given symbol and interval. Market might be closed or invalid parameters."
        
        # TwelveData returns latest first, so reverse to have oldest first
        return "success", list(reversed(candles))

    except requests.exceptions.Timeout:
        return "error", "Request timed out. TwelveData API might be slow or unreachable."
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else "Unknown"
        if status_code == 429:
            return "error", "API rate limit exceeded. Please wait before making more requests."
        return "error", f"HTTP error from TwelveData: {e}. Check API key or symbol."
    except requests.exceptions.ConnectionError:
        return "error", "Connection error. Could not reach TwelveData API."
    except requests.exceptions.RequestException as e:
        return "error", f"An unexpected request error occurred: {e}"
    except json.JSONDecodeError:
        return "error", "Failed to decode JSON response from TwelveData."
    except Exception as e:
        return "error", f"An unexpected error occurred during data fetch: {e}"

# === Indicators ===
def ema_series(data, period):
    ema_values = []
    k = 2 / (period + 1)
    ema = data[0]
    ema_values.append(ema)
    for price in data[1:]:
        ema = price * k + ema * (1 - k)
        ema_values.append(ema)
    return ema_values

def calculate_ema(closes, period=9):
    return ema_series(closes, period)[-1]

def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50
    gains, losses = [], []
    for i in range(1, period + 1):
        delta = closes[-i] - closes[-(i + 1)]
        if delta >= 0:
            gains.append(delta)
        else:
            losses.append(abs(delta))
    avg_gain = sum(gains) / period if gains else 0.01
    avg_loss = sum(losses) / period if losses else 0.01
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# NEW: VWAP CALCULATION
def calculate_vwap(candles):
    """Calculates Volume Weighted Average Price (VWAP)"""
    cumulative_volume = 0
    cumulative_price_volume = 0
    
    for candle in candles:
        high = float(candle['high'])
        low = float(candle['low'])
        close = float(candle['close'])
        volume = float(candle.get('volume', 1))  # Default to 1 if volume missing
        
        typical_price = (high + low + close) / 3
        cumulative_price_volume += typical_price * volume
        cumulative_volume += volume
        
    return cumulative_price_volume / cumulative_volume if cumulative_volume > 0 else 0

# NEW: MACD CALCULATION
def calculate_macd(closes, fast=12, slow=26, signal=9):
    ema_fast = ema_series(closes, fast)
    ema_slow = ema_series(closes, slow)
    
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = ema_series(macd_line, signal)
    histogram = [m - s for m, s in zip(macd_line, signal_line)]
    
    return macd_line[-1], signal_line[-1], histogram[-1]

def calculate_indicators(candles):
    closes = [float(c['close']) for c in candles]  # latest candle nasa dulo
    highs = [float(c['high']) for c in candles]
    lows = [float(c['low']) for c in candles]
    
    vwap = calculate_vwap(candles)
    macd_line, macd_signal, macd_hist = calculate_macd(closes)
    
    return {
        "MA": round(sum(closes) / len(closes), 4),
        "EMA": round(calculate_ema(closes, 9), 4),
        "RSI": round(calculate_rsi(closes), 2),
        "Resistance": round(max(highs), 4),
        "Support": round(min(lows), 4),
        "VWAP": round(vwap, 4),
        "MACD_LINE": round(macd_line, 4),
        "MACD_SIGNAL": round(macd_signal, 4),
        "MACD_HIST": round(macd_hist, 4)
    }

# === Telegram Handlers ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /start command."""
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

    # Get user info from database
    user_info = get_user_info(user_id)
    
    # Update last active timestamp
    save_user_info(user_id)
    
    if user_info and user_info.get('api_key') and user_info.get('agreed_to_disclaimer'):
        # User has already completed setup, show pair selection
        kb = []
        for i in range(0, len(PAIRS), 3): 
            row_buttons = [InlineKeyboardButton(get_flagged_pair_name(PAIRS[j]), callback_data=f"pair|{PAIRS[j]}") 
                        for j in range(i, min(i+3, len(PAIRS)))]
            kb.append(row_buttons)

        await update.message.reply_text("🔑 API key loaded.\n💱 Choose Pair:", reply_markup=InlineKeyboardMarkup(kb))
        return

    # Check if user has agreed to disclaimer but hasn't set API key
    if user_info and user_info.get('agreed_to_disclaimer'):
        await update.message.reply_text("🔐 Please enter your API key:")
        return

    # First time user, show disclaimer
    kb = [[InlineKeyboardButton("✅ I Understand", callback_data="agree_disclaimer")]]
    await update.message.reply_text(
        "⚠️ DISCLAIMER\nThis bot provides educational signals only.\nYou are the engine of your prosperity. By using this bot, you agree to manage your risk wisely, stay disciplined, keep learning, and accept full responsibility for your trading journey.",
        reply_markup=InlineKeyboardMarkup(kb)
    )

async def check_joined_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles callback for checking channel membership."""
    query = update.callback_query
    user_id = query.from_user.id
    chat_id = query.message.chat_id
    await query.answer()

    if query.data == "check_joined":
        if await is_user_joined(user_id, context.bot):
            try:
                await query.message.delete()
            except Exception as e:
                logger.warning(f"Could not delete message for user {user_id} in check_joined_callback: {e}")
            
            # Get user info from database
            user_info = get_user_info(user_id)
            
            # Update last active timestamp
            save_user_info(user_id)
            
            if user_info and user_info.get('api_key') and user_info.get('agreed_to_disclaimer'):
                # User has already completed setup, show pair selection
                kb = []
                for i in range(0, len(PAIRS), 3): 
                    row_buttons = [InlineKeyboardButton(get_flagged_pair_name(PAIRS[j]), callback_data=f"pair|{PAIRS[j]}") 
                                for j in range(i, min(i+3, len(PAIRS)))]
                    kb.append(row_buttons)

                await context.bot.send_message(chat_id, "🔑 API key loaded.\n💱 Choose Pair:", reply_markup=InlineKeyboardMarkup(kb))
                return

            # Check if user has agreed to disclaimer but hasn't set API key
            if user_info and user_info.get('agreed_to_disclaimer'):
                await context.bot.send_message(chat_id, "🔐 Please enter your API key:")
                return

            # First time user, show disclaimer
            kb = [[InlineKeyboardButton("✅ I Understand", callback_data="agree_disclaimer")]]
            await context.bot.send_message(
                chat_id,
                "⚠️ DISCLAIMER\nThis bot provides educational signals only.\nYou are the engine of your prosperity. By using this bot, you agree to manage your risk wisely, stay disciplined, keep learning, and accept full responsibility for your trading journey.",
                reply_markup=InlineKeyboardMarkup(kb)
            )
        else:
            await query.answer("❗ You still haven't joined the channel. Please join and then click the button again.", show_alert=True)

async def howto(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provides instructions on how to use the bot."""
    reminder = await get_friendly_reminder()
    await update.message.reply_text(reminder, parse_mode='Markdown')

async def get_friendly_reminder() -> str:
    """Returns the formatted how-to message."""
    return (
        "📌 *Welcome to YSBONG TRADER™ – Friendly Reminder* 💬\n\n"
        "Hello Trader 👋\n\n"
        "Here’s how to get started with your *real live signals* (not simulation or OTC):\n\n"
        "🧑‍🏫 *How to Use the Bot*\n"
        "1. 🔑 Get your API key from https://twelvedata.com\n"
        "   → Register, log in, dashboard > API Key\n"
        "2. Copy your API KEY. Please keep your API key safe — do not share it with anyone... \n"
        "3.Return to the bot || Tap the menu button || Tap start\n"
        "4. ✅ Agree to the Disclaimer\n"   
        "   → Paste it here in the bot\n"
        "5. 💱 Choose Trading Pair & Timeframe\n"
        "6. ⚡ Click 📶 GET SIGNAL\n\n"
        "📢 *Note:*\n"
        "🏦 This is not OTC. Signals are based on real market data using your API key.\n"
        "🧠 Results depend on live charts.\n\n"
        "⚠️ *No trading on weekends* - the market is closed for non-OTC assets.\n"
        "🙇 *Beginners:*\n"
        "🧑‍💻  Practice first — observe signals.\n"
        "👉 Register here: https://pocket-friends.com/r/w2enb3tukw\n"
        "💵 Deposit when you're confident (min $10).\n\n"
        
        " 🔑 *About TwelveData API Key*\n" 

        "YSBONG TRADER™ uses real-time market data powered by [TwelveData](https://twelvedata.com).\n"
        "You’ll need an API key to activate signals.\n"
        "🆓 **Free Tier (Default when you register)** \n"
        "- ⏱️ Up to 800 API calls per day\n"
        "- 🔄 Max 8 requests per minute\n\n"

        "✌️✌️ GOOD LUCK TRADER ✌️✌️\n\n"

        "🤗 *Be patient. Be disciplined.*\n"
        "😋 *Greedy traders don't last-the market eats them alive.*\n"
        "Respect the market.\n"
        "– *YSBONG TRADER™ powered by PROSPERITY ENGINES™* 💪"
    )

async def disclaimer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Displays the financial risk disclaimer."""
    disclaimer_msg = (
    "⚠️ *Financial Risk Disclaimer*\n\n"
    "Trading involves real risk. This bot provides educational signals only.\n"
    "*Not financial advice.*\n\n"
    "🤔 Be wise. Only trade what you can afford to lose.\n"
    "🎯 Results depend on your discipline, not predictions.\n\n"
    "☣️ *Avoid overtrading!* More trades don’t mean more profits — they usually mean more mistakes.\n"
    "⏳🤚 Wait for clean setups, and trust the process.\n"
)
    await update.message.reply_text(disclaimer_msg, parse_mode='Markdown')

async def handle_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles inline keyboard button presses."""
    query = update.callback_query
    user_id = query.from_user.id
    chat_id = query.message.chat_id
    await query.answer()
    
    try:
        await query.message.delete()
    except Exception as e:
        logger.warning(f"Could not delete message for user {user_id} in handle_buttons: {e}")
        pass

    data = query.data
    if data == "agree_disclaimer":
        # Mark user as having agreed to disclaimer
        save_user_info(user_id, agreed_to_disclaimer=True)
        await context.bot.send_message(chat_id, "🔐 Please enter your API key:")
    elif data.startswith("pair|"):
        pair = data.split("|")[1]
        save_user_info(user_id, current_pair=pair)
        
        half = len(TIMEFRAMES) // 2
        kb = [
            [InlineKeyboardButton(tf, callback_data=f"timeframe|{tf}") for tf in TIMEFRAMES[:half]],
            [InlineKeyboardButton(tf, callback_data=f"timeframe|{tf}") for tf in TIMEFRAMES[half:]]
        ]
        await context.bot.send_message(chat_id, "⏰ Choose Timeframe:", reply_markup=InlineKeyboardMarkup(kb))
    elif data.startswith("timeframe|"):
        timeframe = data.split("|")[1]
        save_user_info(user_id, current_timeframe=timeframe)
        
        await context.bot.send_message(
            chat_id,
            "✅ Ready to generate signal!",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📶 GET SIGNAL 📶", callback_data="get_signal")]])
        )
    elif data == "get_signal":
        await generate_signal(update, context)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming text messages, primarily for API key input."""
    user_id = update.effective_user.id
    text = update.message.text.strip()
    chat_id = update.effective_chat.id

    # Check if user has agreed to disclaimer but hasn't set API key
    user_info = get_user_info(user_id)
    if user_info and user_info.get('agreed_to_disclaimer') and not user_info.get('api_key'):
        # Save API key
        save_user_info(user_id, api_key=text)
        
        kb = []
        for i in range(0, len(PAIRS), 3): 
            row_buttons = [InlineKeyboardButton(get_flagged_pair_name(PAIRS[j]), callback_data=f"pair|{PAIRS[j]}") 
                           for j in range(i, min(i + 3, len(PAIRS)))]
            kb.append(row_buttons)
        await context.bot.send_message(chat_id, "🔐 API Key saved.\n💱 Choose Currency Pair:", reply_markup=InlineKeyboardMarkup(kb))

# Enhanced Signal Generation with Decorator
@smart_signal_strategy
async def generate_signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generates trading signals using enhanced strategy"""
    pass

def store_signal(user_id: int, pair: str, tf: str, action: str, price: float, indicators: Dict) -> None:
    """Stores a generated signal into the database."""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return
            
        with conn.cursor() as cur:
            cur.execute('''
                INSERT INTO signals (user_id, pair, timeframe, action_for_db, price, rsi, ema, ma, resistance, support, 
                                     vwap, macd_line, macd_signal, macd_hist)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (user_id, pair, tf, action, price, indicators["RSI"], indicators["EMA"], indicators["MA"], indicators["Resistance"], indicators["Support"], 
                  indicators["VWAP"], indicators["MACD_LINE"], indicators["MACD_SIGNAL"], indicators["MACD_HIST"]))
            conn.commit()
    except Exception as e:
        logger.error(f"Error storing signal to DB: {e}")
    finally:
        if conn:
            return_db_connection(conn)

async def reset_api(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Resets the user's stored API key."""
    user_id = update.effective_user.id
    
    user_info = get_user_info(user_id)
    if user_info and user_info.get('api_key'):
        remove_user_api_key(user_id)
        await update.message.reply_text("🗑️ API key removed. Please enter your new API key now or use /start to set a new one.")
    else:
        await update.message.reply_text("ℹ️ No API key found to reset.")

async def feedback_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles feedback (win/loss) provided by the user."""
    query = update.callback_query
    user_id = query.from_user.id
    chat_id = query.message.chat_id
    await query.answer()
    
    data = query.data.split('|')
    if data[0] == "feedback":
        feedback_result = data[1]
        
        conn = None
        try:
            conn = get_db_connection()
            if conn is None:
                return
                
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM signals WHERE user_id = %s ORDER BY timestamp DESC LIMIT 1", (user_id,))
                row = cur.fetchone()
                if row:
                    signal_id = row['id']
                    cur.execute("UPDATE signals SET feedback = %s WHERE id = %s", (feedback_result, signal_id))
                    conn.commit()
                    logger.info(f"Feedback saved for signal {signal_id}: {feedback_result}")
                else:
                    logger.warning(f"No previous signal found for user {user_id} to apply feedback.")
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
        finally:
            if conn:
                return_db_connection(conn)

        try:
            await query.edit_message_text(f"✅ Feedback saved: **{feedback_result.upper()}**. Thank you for your input. 😘😘😘!", parse_mode='Markdown')
        except Exception as e:
            logger.warning(f"Could not edit message for feedback for user {user_id}: {e}")
            await context.bot.send_message(chat_id, f"✅ Feedback saved: **{feedback_result.upper()}**. Thank you for your input. 😘😘😘!")

# === New Features ===
INTRO_MESSAGE = """
📢 WELCOME TO YSBONG TRADER™ – SIGNAL SCANNER 📡

✍️ Designed to guide both beginners and experienced traders through real-time market signals.

🫣 What to Expect:
🔄 Auto-generated signals (BUY/SELL)
🕯️ Smart detection from indicators + candle logic
⚡ Fast, clean, no-hype trading alerts
🤖 Hybrid AI Model (RFC+NN Ensemble)

💾 Feedback? Use the Win/Loss buttons  
→ Your result helps improve future signals

👭 Invite your friends to join:
https://t.me/ProsperityEngines

🤓 Trade smart. Stay focused. Respect the charts.
🐲 Let the PROSPERITY ENGINE help you sharpen your instincts.

— YSBONG TRADER™  
“SIGNAL SENT. PROSPERITY LOADED.”
"""

async def intro_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends the bot's introduction message."""
    await update.message.reply_text(INTRO_MESSAGE)

def get_all_users() -> List[int]:
    """Retrieves all unique user IDs from the user_api_keys table."""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return []
            
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT user_id FROM user_api_keys")
            users = [row['user_id'] for row in cur.fetchall()]
        return users
    except Exception as e:
        logger.error(f"Error fetching all users: {e}")
        return []
    finally:
        if conn:
            return_db_connection(conn)

async def send_intro_to_all_users(app: ApplicationBuilder) -> None:
    """Sends the introduction message to all users with saved API keys."""
    users = get_all_users()
    if not users:
        logger.info("No users found to send intro message")
        return
        
    logger.info(f"Attempting to send intro message to {len(users)} users...")
    
    for user_id in users:
        try:
            # Check if bot can send messages to this user (e.g., they haven't blocked the bot)
            chat_member = await app.bot.get_chat_member(chat_id=user_id, user_id=app.bot.id)
            if chat_member.status in [ChatMember.KICKED, ChatMember.LEFT]:
                logger.info(f"Skipping intro message for user {user_id}: Bot is blocked or user left chat.")
                continue

            await app.bot.send_message(chat_id=user_id, text=INTRO_MESSAGE)
            logger.info(f"✅ Intro sent to user: {user_id}")
            await asyncio.sleep(0.2) # Small delay to avoid hitting Telegram API limits
        except Exception as e:
            logger.warning(f"❌ Failed to send intro to {user_id}: {e}")

# === Start Bot ===
if __name__ == '__main__':
    # IMPORTANT: Load token from environment variable
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

    if not TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set. Bot cannot start.")
        print("ERROR: TELEGRAM_BOT_TOKEN environment variable not set. Please set it or add it to a .env file.")
        exit(1)

    app = ApplicationBuilder().token(TOKEN).build()

    # Load ML model after logger is ready
    try:
        load_ensemble_model()
        if ensemble_model:
            logger.info("🤖 AI Model Ready: Hybrid Ensemble (RFC+NN)")
        else:
            logger.warning("⚠️ AI Model Failed to Load - Using Rule-Based Fallback")
    except Exception as e:
        logger.error(f"❌ Critical error loading AI model: {e}")
        logger.warning("⚠️ Using rule-based trading strategy only")

    # Add command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("howto", howto))
    app.add_handler(CommandHandler("disclaimer", disclaimer))
    app.add_handler(CommandHandler("resetapikey", reset_api))
    app.add_handler(CommandHandler("intro", intro_command))

    # Add other handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_buttons, pattern="^(pair|timeframe|get_signal|agree_disclaimer).*"))
    app.add_handler(CallbackQueryHandler(feedback_callback_handler, pattern=r"^feedback\|(win|loss)$"))
    app.add_handler(CallbackQueryHandler(check_joined_callback, pattern="^check_joined$"))

    # Setup scheduled tasks
    scheduler = BackgroundScheduler()
    # Schedule to run every Monday at 9 AM local time
    scheduler.add_job(lambda: asyncio.run(send_intro_to_all_users(app)), 'cron', day_of_week='mon', hour=9)
    # Schedule to clean up inactive users every day at 2 AM
    scheduler.add_job(cleanup_inactive_users, 'cron', hour=2)
    scheduler.start()
    logger.info("⏰ Scheduled tasks configured")

    logger.info("✅ YSBONG TRADER™ is LIVE with AI Trading...")
    app.run_polling(drop_pending_updates=True)