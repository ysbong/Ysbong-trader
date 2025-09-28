import os, json, logging, asyncio, requests, sqlite3, time
import numpy as np
from flask import Flask
from threading import Thread
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ChatMember
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)

# === Logging Setup (Critical to be first) ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("⚡ Initializing YSBONG TRADER™ - GOLD SIGNALS EDITION")

# Data Handling Imports
import pandas as pd
from datetime import datetime
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

# === Entry Point Adjustment Function ===
def adjust_entry_point(action, current_price, indicators, atr, timeframe):
    """
    Adjusts entry point to be more conservative/aggressive based on strategy
    """
    # Timeframe-based adjustment multipliers
    tf_adjustments = {
        "M1": 0.3, "M5": 0.4, "M15": 0.5, 
        "M30": 0.6, "H1": 0.7, "H4": 0.8,
        "D1": 0.9, "WEEK": 1.0
    }
    
    adjustment_factor = tf_adjustments.get(timeframe, 0.5)
    atr_adjustment = atr * 0.1 * adjustment_factor  # 10% of ATR
    
    if action == "BUY":
        # For BUY: Adjust entry LOWER than current price (better entry)
        support_distance = current_price - indicators["Support"]
        resistance_distance = indicators["Resistance"] - current_price
        
        # Method 1: Pullback towards support
        if support_distance < resistance_distance:
            # Closer to support, be more aggressive
            adjustment = min(atr_adjustment, support_distance * 0.3)
        else:
            # Further from support, be conservative
            adjustment = atr_adjustment * 0.5
            
        adjusted_entry = current_price - adjustment
        
        # Method 2: Use recent low as reference
        recent_low = indicators["Support"]
        if current_price - recent_low > atr * 0.5:
            # If far from recent low, aim for pullback
            adjusted_entry = max(adjusted_entry, recent_low + atr * 0.2)
        
    else:  # SELL
        # For SELL: Adjust entry HIGHER than current price (better entry)
        resistance_distance = indicators["Resistance"] - current_price
        support_distance = current_price - indicators["Support"]
        
        # Method 1: Rally towards resistance
        if resistance_distance < support_distance:
            # Closer to resistance, be more aggressive
            adjustment = min(atr_adjustment, resistance_distance * 0.3)
        else:
            # Further from resistance, be conservative
            adjustment = atr_adjustment * 0.5
            
        adjusted_entry = current_price + adjustment
        
        # Method 2: Use recent high as reference
        recent_high = indicators["Resistance"]
        if recent_high - current_price > atr * 0.5:
            # If far from recent high, aim for rally
            adjusted_entry = min(adjusted_entry, recent_high - atr * 0.2)
    
    # Ensure adjustment is reasonable (not too far from current price)
    max_adjustment = current_price * 0.005  # Max 0.5% adjustment
    if action == "BUY":
        adjusted_entry = max(adjusted_entry, current_price - max_adjustment)
    else:
        adjusted_entry = min(adjusted_entry, current_price + max_adjustment)
    
    return round(adjusted_entry, 2)

def adjust_entry_fixed(action, current_price, fixed_offset_pips=5.0):
    """
    Adjust entry by fixed pip offset
    """
    pip_value = 0.01  # Adjust based on your instrument's pip size
    
    if action == "BUY":
        return current_price - (fixed_offset_pips * pip_value)
    else:  # SELL
        return current_price + (fixed_offset_pips * pip_value)

# === Smart Signal Decorator ===
def smart_signal_strategy(func: Callable) -> Callable:
    """Decorator to enhance signal generation with advanced strategies"""
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        query = update.callback_query
        user_id = query.from_user.id
        chat_id = query.message.chat_id
        
        # Get user data
        data = user_data.get(user_id, {})
        pair = "XAU/USD"  # Fixed to XAU/USD only
        tf = data.get("timeframe", "M1")
        api_key = data.get("api_key")

        if not api_key:
            await context.bot.send_message(chat_id, text="❌ API key not found. Please set your API key using /start.")
            return

        # Convert timeframe to interval format
        def timeframe_to_interval(tf):
            mapping = {
                "M1": "1min", "M5": "5min", "M15": "15min", 
                "M30": "30min", "H1": "1h", "H4": "4h",
                "D1": "1day", "WEEK": "1week"
            }
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
        loading_msg = await context.bot.send_message(chat_id, text=f"🔍 Analyzing GOLD market... {loading_frames[0]}")
        
        # Animate loading bar
        for i in range(1, len(loading_frames)):
            await asyncio.sleep(0.1)
            try:
                await loading_msg.edit_text(text=f"🔍 Analyzing GOLD market... {loading_frames[i]}")
            except Exception as e:
                logger.error(f"Error editing loading message: {e}")
                break

        # Fetch data
        status, result = fetch_data(api_key, "XAU/USD", interval=timeframe_to_interval(tf))
        if status == "error":
            try:
                await loading_msg.delete()
            except:
                pass
            user_data[user_id].pop("api_key", None)
            user_data[user_id]["step"] = "awaiting_api"
            await context.bot.send_message(chat_id, text=f"❌ {result}")
            return

        # Calculate indicators
        try:
            indicators = calculate_indicators(result)
            
            # 🔥 CRITICAL FIX: Get the LATEST (most recent) candle price, not the oldest!
            current_price = float(result[-1]["close"])  # FIXED: Changed from result[0] to result[-1]
            
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
            
        # Calculate trading parameters
        atr = calculate_atr(result)
        
        # Calculate adjusted entry point (NEW FEATURE)
        adjusted_entry = adjust_entry_point(
            action_for_db, current_price, indicators, atr, tf
        )
        
        # Use adjusted entry for risk management
        stop_loss, take_profit = calculate_risk_management(
            action_for_db, adjusted_entry, indicators["Support"], 
            indicators["Resistance"], atr, tf
        )
        
        # Determine market trend
        market_trend = determine_market_trend(
            current_price, indicators["EMA"], indicators["MACD_HIST"], 
            indicators["RSI"]
        )

        # Format and send signal
        flagged_pair = "🥇XAU/USD"  # Gold-specific display
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        signal = (
            "🚀 *YSBONG TRADER™ GOLD SIGNAL*\n"
            f"━━━━━━━━━━━━━━━━━━━\n" 
            f"💹 PAIR: {flagged_pair}\n"
            f"⏱ TIMEFRAME: {tf}\n"
            f"🧨 DIRECTION: {action}\n"
            f"🎯 CONFIDENCE: {confidence_level}\n"
            f"📊 MARKET TREND: {market_trend}\n"
            f"⏰ TIME: {current_time}\n"
            f"━━━━━━━━━━━━━━━━━━━\n" 
            f"💰 CURRENT PRICE: ${current_price:.2f}\n"
            f"🎯 ADJUSTED ENTRY: ${adjusted_entry:.2f}\n"
            f"🛡️ STOP LOSS: ${stop_loss:.2f}\n"
            f"🎯 TAKE PROFIT: ${take_profit:.2f}\n"
            f"📊 RISK/REWARD: 1:{abs((take_profit - adjusted_entry) / (adjusted_entry - stop_loss)):.1f}\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"📊 *GOLD ANALYSIS*\n"
            f"💰 Price: ${current_price:.2f}\n"
            f"📉 RSI: {indicators['RSI']:.1f} ({'Overbought' if indicators['RSI'] > 70 else 'Oversold' if indicators['RSI'] < 30 else 'Neutral'})\n"
            f"📈 EMA: ${indicators['EMA']:.2f}\n"
            f"📊 VWAP: ${indicators['VWAP']:.2f}\n"
            f"📈 MACD: {indicators['MACD_HIST']:.4f} ({'Bullish' if indicators['MACD_HIST'] > 0 else 'Bearish'})\n"
            f"🛡️ Support: ${indicators['Support']:.2f}\n"
            f"🚧 Resistance: ${indicators['Resistance']:.2f}\n"
            f"📈 ATR (Volatility): ${atr:.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"🤖 *AI Model Used:* Hybrid Ensemble (RFC+NN)\n"
            f"🎯 *Entry Strategy:* Adjusted for better risk/reward\n"
            f"☣️ Gold is volatile! Manage risk carefully...\n"
            f"⚠️ Always use proper risk management (1-2% per trade)\n"
        )
        
        # Delete loading message before sending signal
        try:
            await loading_msg.delete()
        except Exception as e:
            logger.warning(f"Failed to delete loading message: {e}")
        
        await context.bot.send_message(chat_id=chat_id, text=signal, parse_mode='Markdown')
        
        # Store the signal
        store_signal(user_id, "XAU/USD", tf, action_for_db, current_price, indicators)

    return wrapper

# === Improved Risk Management Functions ===
def calculate_atr(candles, period=14):
    """Calculate Average True Range for volatility measurement"""
    true_ranges = []
    
    for i in range(1, len(candles)):
        high = float(candles[i]['high'])
        low = float(candles[i]['low'])
        prev_close = float(candles[i-1]['close'])
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = max(tr1, tr2, tr3)
        true_ranges.append(true_range)
    
    # Calculate ATR as the average of true ranges
    if len(true_ranges) >= period:
        atr = sum(true_ranges[-period:]) / period
    else:
        atr = sum(true_ranges) / len(true_ranges) if true_ranges else 0
    
    return atr

def calculate_risk_management(action, entry_price, support, resistance, atr, timeframe):
    """
    Calculate optimized stop loss and take profit levels based on:
    - Market volatility (ATR)
    - Timeframe sensitivity
    - Support/resistance levels
    - Optimal risk-reward ratios
    """
    # Timeframe-based multipliers (shorter timeframes = tighter stops)
    tf_multipliers = {
        "M1": 0.8, "M5": 1.0, "M15": 1.2, 
        "M30": 1.5, "H1": 2.0, "H4": 2.5,
        "D1": 3.0, "WEEK": 4.0
    }
    
    tf_multiplier = tf_multipliers.get(timeframe, 1.0)
    
    # Base ATR multiplier adjusted for timeframe
    base_atr_multiplier = 1.2 * tf_multiplier
    
    if action == "BUY":
        # Calculate stop loss using multiple methods and choose the best
        sl_methods = []
        
        # Method 1: ATR-based stop loss
        sl_atr = entry_price - (atr * base_atr_multiplier)
        sl_methods.append(sl_atr)
        
        # Method 2: Support-based stop loss (with buffer)
        sl_support = support * 0.995  # 0.5% below support
        sl_methods.append(sl_support)
        
        # Method 3: Fixed percentage stop (0.3% for shorter timeframes, 0.5% for longer)
        fixed_percent = 0.003 if timeframe in ["M1", "M5", "M15"] else 0.005
        sl_fixed = entry_price * (1 - fixed_percent)
        sl_methods.append(sl_fixed)
        
        # Choose the most conservative (highest) stop loss for BUY (provides more room)
        stop_loss = max(sl_methods)
        
        # Ensure stop loss is reasonable (not too far)
        max_sl_distance = entry_price * 0.02  # Max 2% stop loss
        stop_loss = max(stop_loss, entry_price - max_sl_distance)
        
        # Calculate take profit with optimal risk-reward ratio
        risk_amount = entry_price - stop_loss
        reward_ratios = {
            "M1": 1.5, "M5": 1.8, "M15": 2.0,
            "M30": 2.2, "H1": 2.5, "H4": 3.0,
            "D1": 3.5, "WEEK": 4.0
        }
        
        reward_ratio = reward_ratios.get(timeframe, 2.0)
        take_profit = entry_price + (risk_amount * reward_ratio)
        
        # Cap take profit at resistance level with buffer
        max_tp = resistance * 0.995
        take_profit = min(take_profit, max_tp)
        
    else:  # SELL
        # Calculate stop loss using multiple methods
        sl_methods = []
        
        # Method 1: ATR-based stop loss
        sl_atr = entry_price + (atr * base_atr_multiplier)
        sl_methods.append(sl_atr)
        
        # Method 2: Resistance-based stop loss (with buffer)
        sl_resistance = resistance * 1.005  # 0.5% above resistance
        sl_methods.append(sl_resistance)
        
        # Method 3: Fixed percentage stop
        fixed_percent = 0.003 if timeframe in ["M1", "M5", "M15"] else 0.005
        sl_fixed = entry_price * (1 + fixed_percent)
        sl_methods.append(sl_fixed)
        
        # Choose the most conservative (lowest) stop loss for SELL
        stop_loss = min(sl_methods)
        
        # Ensure stop loss is reasonable
        max_sl_distance = entry_price * 0.02  # Max 2% stop loss
        stop_loss = min(stop_loss, entry_price + max_sl_distance)
        
        # Calculate take profit with optimal risk-reward ratio
        risk_amount = stop_loss - entry_price
        reward_ratios = {
            "M1": 1.5, "M5": 1.8, "M15": 2.0,
            "M30": 2.2, "H1": 2.5, "H4": 3.0,
            "D1": 3.5, "WEEK": 4.0
        }
        
        reward_ratio = reward_ratios.get(timeframe, 2.0)
        take_profit = entry_price - (risk_amount * reward_ratio)
        
        # Cap take profit at support level with buffer
        min_tp = support * 1.005
        take_profit = max(take_profit, min_tp)
    
    # Final validation to ensure reasonable gaps
    min_gap = entry_price * 0.001  # Minimum 0.1% gap
    max_gap = entry_price * 0.03   # Maximum 3% gap
    
    if action == "BUY":
        sl_gap = entry_price - stop_loss
        tp_gap = take_profit - entry_price
        
        if sl_gap < min_gap:
            stop_loss = entry_price - min_gap
        elif sl_gap > max_gap:
            stop_loss = entry_price - max_gap
            
        if tp_gap < min_gap:
            take_profit = entry_price + min_gap
        elif tp_gap > max_gap:
            take_profit = entry_price + max_gap
    else:  # SELL
        sl_gap = stop_loss - entry_price
        tp_gap = entry_price - take_profit
        
        if sl_gap < min_gap:
            stop_loss = entry_price + min_gap
        elif sl_gap > max_gap:
            stop_loss = entry_price + max_gap
            
        if tp_gap < min_gap:
            take_profit = entry_price - min_gap
        elif tp_gap > max_gap:
            take_profit = entry_price - max_gap
    
    return round(stop_loss, 2), round(take_profit, 2)

def determine_market_trend(price, ema, macd_hist, rsi):
    """Determine the overall market trend based on multiple indicators"""
    trend_score = 0
    
    # Price vs EMA (trend direction)
    if price > ema:
        trend_score += 1
    else:
        trend_score -= 1
        
    # MACD histogram (momentum)
    if macd_hist > 0:
        trend_score += 1
    else:
        trend_score -= 1
        
    # RSI (momentum strength)
    if rsi > 55:
        trend_score += 1
    elif rsi < 45:
        trend_score -= 1
        
    # Determine trend based on score
    if trend_score >= 2:
        return "STRONG BULLISH 📈"
    elif trend_score >= 1:
        return "BULLISH 📈"
    elif trend_score <= -2:
        return "STRONG BEARISH 📉"
    elif trend_score <= -1:
        return "BEARISH 📉"
    else:
        return "NEUTRAL ➡️"

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
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            query = """
            SELECT 
                MA, EMA, RSI, Resistance, Support, VWAP, 
                macd_line, macd_signal, macd_hist,
                price,
                action_for_db
            FROM signals
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
    return "YSBONG TRADER™ GOLD SIGNALS (Active) is awake and scanning!"

@web_app.route("/health")
def health() -> Tuple[str, int]:
    """Health check endpoint."""
    return "✅ YSBONG™ GOLD SIGNALS is alive and kicking!", 200

def run_web() -> None:
    """Runs the Flask web application in a separate thread."""
    port = int(os.environ.get("PORT", 8080))
    web_app.run(host="0.0.0.0", port=port)

# Start the Flask app in a separate thread
Thread(target=run_web).start()

# === SQLite Learning Memory ===
DB_FILE = "ysbong_memory.db"
SQLITE_TIMEOUT = 15.0 # seconds

def init_db() -> None:
    """Initializes the SQLite database tables."""
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS user_api_keys (
                    user_id INTEGER PRIMARY KEY,
                    api_key TEXT NOT NULL
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS candle_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    candle_data TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
    except sqlite3.Error as e:
        logging.error(f"SQLite initialization error: {e}")

init_db()

user_data: dict = {}
usage_count: dict = {}

def load_saved_keys() -> dict:
    """Loads saved API keys from the database."""
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute("SELECT user_id, api_key FROM user_api_keys")
            keys = {str(row[0]): row[1] for row in c.fetchall()}
            return keys
    except sqlite3.Error as e:
        logger.error(f"Error loading API keys from DB: {e}")
        return {}

def save_keys(user_id: int, api_key: str) -> None:
    """Saves an API key to the database."""
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO user_api_keys (user_id, api_key) VALUES (?, ?)", (user_id, api_key))
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error saving API key to DB: {e}")

def remove_key(user_id: int) -> None:
    """Removes an API key from the database."""
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM user_api_keys WHERE user_id = ?", (user_id,))
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error removing API key from DB: {e}")

saved_keys: dict = load_saved_keys() # Initial load

# === Constants ===
TIMEFRAMES: List[str] = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "WEEK"]

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
        # This is correct for indicator calculations (they need chronological order)
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
    closes = [float(c['close']) for c in candles]  # oldest to newest
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

    # If user has joined, proceed with normal start flow
    user_data[user_id] = {}
    usage_count[user_id] = usage_count.get(user_id, 0)
    
    api_key_from_db = load_saved_keys().get(str(user_id))

    if api_key_from_db:
        user_data[user_id]["api_key"] = api_key_from_db
        # Skip pair selection and go directly to timeframe selection for XAU/USD
        half = len(TIMEFRAMES) // 2
        kb = [
            [InlineKeyboardButton(tf, callback_data=f"timeframe|{tf}") for tf in TIMEFRAMES[:half]],
            [InlineKeyboardButton(tf, callback_data=f"timeframe|{tf}") for tf in TIMEFRAMES[half:]]
        ]
        await update.message.reply_text("🔑 API key loaded.\n⏰ Choose Timeframe:", reply_markup=InlineKeyboardMarkup(kb))
        return

    kb = [[InlineKeyboardButton("✅ I Understand", callback_data="agree_disclaimer")]]
    await update.message.reply_text(
        "⚠️ DISCLAIMER\nThis bot provides educational signals only for GOLD (XAU/USD).\nYou are the engine of your prosperity.",
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
            
            # Proceed with normal start flow
            user_data[user_id] = {}
            usage_count[user_id] = usage_count.get(user_id, 0)
            
            api_key_from_db = load_saved_keys().get(str(user_id))

            if api_key_from_db:
                user_data[user_id]["api_key"] = api_key_from_db
                # Skip pair selection and go directly to timeframe selection for XAU/USD
                half = len(TIMEFRAMES) // 2
                kb = [
                    [InlineKeyboardButton(tf, callback_data=f"timeframe|{tf}") for tf in TIMEFRAMES[:half]],
                    [InlineKeyboardButton(tf, callback_data=f"timeframe|{tf}") for tf in TIMEFRAMES[half:]]
                ]
                await context.bot.send_message(chat_id, "🔑 API key loaded.\n⏰ Choose Timeframe:", reply_markup=InlineKeyboardMarkup(kb))
                return

            kb = [[InlineKeyboardButton("✅ I Understand", callback_data="agree_disclaimer")]]
            await context.bot.send_message(
                chat_id,
                "⚠️ DISCLAIMER\nThis bot provides educational signals only for GOLD (XAU/USD).\nYou are the engine of your prosperity.",
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
        "📌 *Welcome to YSBONG TRADER™ GOLD SIGNALS – Friendly Reminder* 💬\n\n"
        "Hello Gold Trader 👋\n\n"
        "Here's how to get started with your *real live GOLD signals*:\n\n"
        "🧑‍🏫 *How to Use the Bot*\n"
        "1. 🔑 Get your API key from https://twelvedata.com\n"
        "   → Register, log in, dashboard > API Key\n"
        "2. Copy your API KEY || Return to the bot\n"
        "3. Tap the menu button || Tap start\n"
        "4. ✅ Agree to the Disclaimer\n"   
        "   → Paste it here in the bot\n"
        "5. ⏰ Choose Timeframe\n"
        "6. ⚡ Click 📶 GET GOLD SIGNAL\n\n"
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
        "You'll need an API key to activate signals.\n"
        "🆓 **Free Tier (Default when you register)** \n"
        "- ⏱️ Up to 800 API calls per day\n"
        "- 🔄 Max 8 requests per minute\n\n"

        "✌️✌️ GOOD LUCK GOLD TRADER ✌️✌️\n\n"

        "🤗 *Be patient. Be disciplined.*\n"
        "😋 *Greedy traders don't last-the market eats them alive.*\n"
        "Respect the market.\n"
        "– *YSBONG TRADER™ GOLD SIGNALS powered by PROSPERITY ENGINES™* 💪"
    )

async def disclaimer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Displays the financial risk disclaimer."""
    disclaimer_msg = (
    "⚠️ *Financial Risk Disclaimer*\n\n"
    "Trading GOLD involves real risk. This bot provides educational signals only.\n"
    "*Not financial advice.*\n\n"
    "🤔 Be wise. Only trade what you can afford to lose.\n"
    "🎯 Results depend on your discipline, not predictions.\n\n"
    "☣️ *Gold is volatile!* Manage risk carefully with proper position sizing.\n"
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
        await context.bot.send_message(chat_id, "🔐 Please enter your API key:")
        user_data[user_id] = {"step": "awaiting_api"}
    elif data.startswith("timeframe|"):
        user_data[user_id]["timeframe"] = data.split("|")[1]
        await context.bot.send_message(
            chat_id,
            "✅ Ready to generate GOLD signal!",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📶 GET GOLD SIGNAL 📶", callback_data="get_signal")]])
        )
    elif data == "get_signal":
        await generate_signal(update, context)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming text messages, primarily for API key input."""
    user_id = update.effective_user.id
    text = update.message.text.strip()
    chat_id = update.effective_chat.id

    if user_data.get(user_id, {}).get("step") == "awaiting_api":
        user_data[user_id]["api_key"] = text
        user_data[user_id]["step"] = None
        save_keys(user_id, text)
        # Skip pair selection and go directly to timeframe selection for XAU/USD
        half = len(TIMEFRAMES) // 2
        kb = [
            [InlineKeyboardButton(tf, callback_data=f"timeframe|{tf}") for tf in TIMEFRAMES[:half]],
            [InlineKeyboardButton(tf, callback_data=f"timeframe|{tf}") for tf in TIMEFRAMES[half:]]
        ]
        await context.bot.send_message(chat_id, "🔐 API Key saved.\n⏰ Choose Timeframe:", reply_markup=InlineKeyboardMarkup(kb))

# Enhanced Signal Generation with Decorator
@smart_signal_strategy
async def generate_signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generates trading signals using enhanced strategy"""
    pass

def store_signal(user_id: int, pair: str, tf: str, action: str, price: float, indicators: Dict) -> None:
    """Stores a generated signal into the database."""
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO signals (user_id, pair, timeframe, action_for_db, price, rsi, ema, ma, resistance, support, 
                                     vwap, macd_line, macd_signal, macd_hist)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, pair, tf, action, price, indicators["RSI"], indicators["EMA"], indicators["MA"], indicators["Resistance"], indicators["Support"], 
                  indicators["VWAP"], indicators["MACD_LINE"], indicators["MACD_SIGNAL"], indicators["MACD_HIST"]))
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error storing signal to DB: {e}")

async def reset_api(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Resets the user's stored API key."""
    user_id = update.effective_user.id
    
    api_key_exists = load_saved_keys().get(str(user_id))

    if api_key_exists:
        remove_key(user_id)
        if user_id in user_data:
            user_data[user_id].pop("api_key", None)
            user_data[user_id]["step"] = "awaiting_api"
        await update.message.reply_text("🗑️ API key removed. Please enter your new API key now or use /start to set a new one.")
    else:
        await update.message.reply_text("ℹ️ No API key found to reset.")

# === New Features ===
INTRO_MESSAGE = """
📢 WELCOME TO YSBONG TRADER™ – GOLD SIGNAL SCANNER 📡

✍️ Designed to guide both beginners and experienced traders through real-time GOLD signals.

🫣 What to Expect:
🔄 Auto-generated GOLD signals (BUY/SELL)
🕯️ Smart detection from indicators + candle logic
⚡ Fast, clean, no-hype GOLD trading alerts
🤖 Hybrid AI Model (RFC+NN Ensemble)
🎯 **NEW: Adjusted Entry Points** for better risk/reward

👭 Invite your friends to join:
https://t.me/ProsperityEngines

🤓 Trade smart. Stay focused. Respect the charts.
🐲 Let the PROSPERITY ENGINE help you sharpen your GOLD trading instincts.

— YSBONG TRADER™ GOLD SIGNALS  
"GOLD SIGNAL SENT. PROSPERITY LOADED."
"""

async def intro_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends the bot's introduction message."""
    await update.message.reply_text(INTRO_MESSAGE)

def get_all_users() -> List[int]:
    """Retrieves all unique user IDs from the user_api_keys table."""
    try:
        with sqlite3.connect(DB_FILE, timeout=SQLITE_TIMEOUT) as conn:
            c = conn.cursor()
            c.execute("SELECT DISTINCT user_id FROM user_api_keys")
            users = [row[0] for row in c.fetchall()]
        return users
    except sqlite3.Error as e:
        logger.error(f"Error fetching all users: {e}")
        return []

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
    app.add_handler(CallbackQueryHandler(handle_buttons, pattern="^(timeframe|get_signal|agree_disclaimer).*"))
    app.add_handler(CallbackQueryHandler(check_joined_callback, pattern="^check_joined$"))

    # Setup scheduled intro message
    scheduler = BackgroundScheduler()
    # Schedule to run every Monday at 9 AM local time (adjust as needed for server timezone)
    scheduler.add_job(lambda: asyncio.run(send_intro_to_all_users(app)), 'cron', day_of_week='mon', hour=9)
    scheduler.start()
    logger.info("⏰ Scheduled weekly intro message configured (Mondays at 9 AM)")

    logger.info("✅ YSBONG TRADER™ GOLD SIGNALS is LIVE with AI Trading & Adjusted Entry Points...")
    app.run_polling(drop_pending_updates=True)