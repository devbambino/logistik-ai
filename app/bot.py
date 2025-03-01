import os
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from sqlalchemy.orm import Session
from datetime import datetime
import logging

from app.database import SessionLocal
from app.models import User, Trip, StatusUpdate, Location, Issue, Notification
from app.schemas import UserRole, TripStatus
from app.agents.driver_agent import DriverAgent
from app.agents.manager_agent import ManagerAgent
from app.agents.shipper_agent import ShipperAgent
from app.agents.consignee_agent import ConsigneeAgent

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# User session storage
user_sessions = {}

# Quick reply keyboards
driver_keyboard = ReplyKeyboardMarkup([
    [KeyboardButton("At Pickup"), KeyboardButton("Loading"), KeyboardButton("Departed")],
    [KeyboardButton("On Schedule"), KeyboardButton("Slight Delay"), KeyboardButton("Major Delay")],
    [KeyboardButton("Arrived at Destination"), KeyboardButton("Unloading"), KeyboardButton("Completed Delivery")],
    [KeyboardButton("Report Issue"), KeyboardButton("Share Location"), KeyboardButton("Trip Details")]
], resize_keyboard=True)

# Helper functions
def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

async def get_or_create_user(telegram_user, role=UserRole.DRIVER.value):
    db = get_db()
    user = db.query(User).filter(User.telegram_id == str(telegram_user.id)).first()
    
    if not user:
        user = User(
            telegram_id=str(telegram_user.id),
            username=telegram_user.username,
            first_name=telegram_user.first_name,
            last_name=telegram_user.last_name,
            role=role
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    
    return user

async def notify_stakeholders(trip_id, message):
    db = get_db()
    trip = db.query(Trip).filter(Trip.id == trip_id).first()
    
    if not trip:
        return
    
    # Create notifications for all stakeholders
    stakeholders = []
    if trip.manager_id:
        stakeholders.append(trip.manager_id)
    if trip.shipper_id:
        stakeholders.append(trip.shipper_id)
    if trip.consignee_id:
        stakeholders.append(trip.consignee_id)
    
    for stakeholder_id in stakeholders:
        notification = Notification(
            user_id=stakeholder_id,
            trip_id=trip_id,
            message=message
        )
        db.add(notification)
    
    db.commit()

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = await get_or_create_user(update.effective_user)
    
    # Ask for role if not already set
    if not hasattr(context.user_data, "role_set") or not context.user_data["role_set"]:
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("Driver", callback_data="set_role_driver")],
            [InlineKeyboardButton("Manager", callback_data="set_role_manager")],
            [InlineKeyboardButton("Shipper", callback_data="set_role_shipper")],
            [InlineKeyboardButton("Consignee", callback_data="set_role_consignee")]
        ])
        await update.message.reply_text(
            f"Hello {user.first_name}! Welcome to the Logistics AI Bot.\n\n"
            "Please select your role:",
            reply_markup=keyboard
        )
        return
    
    await update.message.reply_text(
        f"Hello {user.first_name}! Welcome to the Logistics AI Bot.\n\n"
        "I'm here to help you manage your logistics operations.\n\n"
        f"You are registered as a {user.role}.\n\n"
        "If you're a driver, you'll receive trip assignments and can update your status.\n"
        "If you're a manager, shipper, or consignee, you'll receive updates about your shipments.",
        reply_markup=driver_keyboard if user.role == UserRole.DRIVER.value else None
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    user = await get_or_create_user(update.effective_user)
    
    if user.role == UserRole.DRIVER.value:
        await update.message.reply_text(
            "Here's how to use this bot as a driver:\n\n"
            "- You'll receive trip assignments with pickup and delivery details\n"
            "- Use the quick reply buttons to update your status\n"
            "- Report issues using the 'Report Issue' button\n"
            "- Share your location when prompted\n"
            "- You can also ask me questions about your trips"
        )
    elif user.role == UserRole.MANAGER.value:
        await update.message.reply_text(
            "Here's how to use this bot as a manager:\n\n"
            "- You can check the status of any shipment using /status [trip_id]\n"
            "- You can ask me about active trips, trip details, and issues\n"
            "- You can send messages to drivers and resolve issues\n"
            "- You can create new trip assignments"
        )
    elif user.role == UserRole.SHIPPER.value:
        await update.message.reply_text(
            "Here's how to use this bot as a shipper:\n\n"
            "- You can check the status of your shipments using /status [trip_id]\n"
            "- You can ask me about your shipments, their status, and driver locations\n"
            "- You can send messages to drivers and managers"
        )
    elif user.role == UserRole.CONSIGNEE.value:
        await update.message.reply_text(
            "Here's how to use this bot as a consignee:\n\n"
            "- You can check the status of incoming shipments using /status [trip_id]\n"
            "- You can ask me about your incoming shipments, their status, and ETAs\n"
            "- You can send messages to drivers and shippers"
        )
    else:
        await update.message.reply_text(
            "Please use /start to set your role first."
        )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check the status of a trip."""
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Please provide a valid trip ID: /status [trip_id]")
        return
    
    trip_id = int(context.args[0])
    db = get_db()
    trip = db.query(Trip).filter(Trip.id == trip_id).first()
    
    if not trip:
        await update.message.reply_text(f"Trip #{trip_id} not found.")
        return
    
    # Get the latest status update
    latest_update = db.query(StatusUpdate).filter(
        StatusUpdate.trip_id == trip_id
    ).order_by(StatusUpdate.created_at.desc()).first()
    
    # Get the latest location
    latest_location = db.query(Location).filter(
        Location.trip_id == trip_id
    ).order_by(Location.timestamp.desc()).first()
    
    response = f"""
    Trip #{trip.id} Status:
    
    Current status: {trip.status}
    Pickup: {trip.pickup_address}
    Delivery: {trip.delivery_address}
    Cargo: {trip.cargo_description}
    """
    
    if latest_update:
        response += f"\nLast update: {latest_update.created_at.strftime('%Y-%m-%d %H:%M')}"
        if latest_update.notes:
            response += f"\nNotes: {latest_update.notes}"
    
    if latest_location:
        response += f"\n\nLast known location: {latest_location.latitude}, {latest_location.longitude}"
        response += f"\nLocation time: {latest_location.timestamp.strftime('%Y-%m-%d %H:%M')}"
    
    await update.message.reply_text(response)

async def set_role_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set the user's role."""
    if not context.args or context.args[0] not in [role.value for role in UserRole]:
        await update.message.reply_text(
            "Please provide a valid role: /set_role [driver|manager|shipper|consignee]"
        )
        return
    
    role = context.args[0]
    user = await get_or_create_user(update.effective_user, role)
    
    # Update the user's role in the database
    db = get_db()
    db_user = db.query(User).filter(User.telegram_id == str(update.effective_user.id)).first()
    if db_user:
        db_user.role = role
        db.commit()
    
    context.user_data["role_set"] = True
    
    await update.message.reply_text(
        f"Your role has been set to {role}. You can now use the bot."
    )
    
    # Send a follow-up message with the appropriate keyboard
    if role == UserRole.DRIVER.value:
        await context.bot.send_message(
            chat_id=update.effective_user.id,
            text="As a driver, you can use these quick reply buttons:",
            reply_markup=driver_keyboard
        )
    else:
        await context.bot.send_message(
            chat_id=update.effective_user.id,
            text=f"As a {role}, you can ask me questions about your shipments and I'll help you manage them."
        )

# Message handlers
async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages from users."""
    user = await get_or_create_user(update.effective_user)
    text = update.message.text
    
    # Handle quick reply buttons for drivers
    if user.role == UserRole.DRIVER.value:
        db = get_db()
        trip = db.query(Trip).filter(
            Trip.driver_id == user.id,
            Trip.status != TripStatus.COMPLETED.value
        ).order_by(Trip.created_at.desc()).first()
        
        if trip:
            if text == "At Pickup":
                await update_trip_status(update, context, trip.id, TripStatus.AT_PICKUP.value)
                await notify_stakeholders(trip.id, f"Driver has arrived at pickup location for Trip #{trip.id}")
                return
            elif text == "Loading":
                await update_trip_status(update, context, trip.id, TripStatus.LOADING.value)
                await notify_stakeholders(trip.id, f"Loading has begun for Trip #{trip.id}")
                return
            elif text == "Departed":
                await update_trip_status(update, context, trip.id, TripStatus.IN_TRANSIT.value)
                await notify_stakeholders(trip.id, f"Driver has departed from pickup location for Trip #{trip.id}")
                return
            elif text == "On Schedule":
                await update_message(update, context, "Thanks for confirming you're on schedule.")
                await notify_stakeholders(trip.id, f"Driver for Trip #{trip.id} reports being on schedule")
                return
            elif text == "Slight Delay":
                await update_trip_status(update, context, trip.id, TripStatus.DELAYED.value, "Slight delay reported")
                await notify_stakeholders(trip.id, f"Driver for Trip #{trip.id} reports a slight delay")
                return
            elif text == "Major Delay":
                await update_trip_status(update, context, trip.id, TripStatus.DELAYED.value, "Major delay reported")
                await notify_stakeholders(trip.id, f"Driver for Trip #{trip.id} reports a major delay")
                return
            elif text == "Arrived at Destination":
                await update_trip_status(update, context, trip.id, TripStatus.AT_DESTINATION.value)
                await notify_stakeholders(trip.id, f"Driver has arrived at destination for Trip #{trip.id}")
                return
            elif text == "Unloading":
                await update_trip_status(update, context, trip.id, TripStatus.UNLOADING.value)
                await notify_stakeholders(trip.id, f"Unloading has begun for Trip #{trip.id}")
                return
            elif text == "Completed Delivery":
                await update_trip_status(update, context, trip.id, TripStatus.COMPLETED.value)
                await notify_stakeholders(trip.id, f"Delivery has been completed for Trip #{trip.id}")
                
                # Send survey
                survey_keyboard = InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton("👍 Good", callback_data=f"survey_good_{trip.id}"),
                        InlineKeyboardButton("👌 OK", callback_data=f"survey_ok_{trip.id}"),
                        InlineKeyboardButton("👎 Bad", callback_data=f"survey_bad_{trip.id}")
                    ]
                ])
                await update.message.reply_text(
                    "Thanks for completing the delivery! How was your experience?",
                    reply_markup=survey_keyboard
                )
                return
            elif text == "Report Issue":
                context.user_data["awaiting_issue"] = trip.id
                await update.message.reply_text(
                    "Please describe the issue you're experiencing:"
                )
                return
            elif text == "Trip Details":
                # Use the driver agent to get trip details
                driver_agent = DriverAgent(db, user.id)
                trip_details = driver_agent._get_current_trip()
                await update.message.reply_text(trip_details)
                return
            elif text == "Share Location":
                await update.message.reply_text(
                    "Please share your current location:",
                    reply_markup=ReplyKeyboardMarkup([
                        [KeyboardButton("Share Location", request_location=True)]
                    ], resize_keyboard=True, one_time_keyboard=True)
                )
                return
    
    # Handle awaiting issue
    if "awaiting_issue" in context.user_data:
        trip_id = context.user_data["awaiting_issue"]
        db = get_db()
        
        # Create issue
        issue = Issue(
            trip_id=trip_id,
            reported_by_id=user.id,
            description=text,
            status="open"
        )
        db.add(issue)
        
        # Update trip status
        trip = db.query(Trip).filter(Trip.id == trip_id).first()
        if trip:
            trip.status = TripStatus.ISSUE_REPORTED.value
            trip.updated_at = datetime.utcnow()
            
            # Create status update
            status_update = StatusUpdate(
                trip_id=trip_id,
                user_id=user.id,
                status=TripStatus.ISSUE_REPORTED.value,
                notes=f"Issue reported: {text}"
            )
            db.add(status_update)
            
            # Notify stakeholders
            await notify_stakeholders(trip_id, f"Issue reported for Trip #{trip_id}: {text}")
        
        db.commit()
        
        del context.user_data["awaiting_issue"]
        
        await update.message.reply_text(
            "Issue reported. Thank you for letting us know.",
            reply_markup=driver_keyboard if user.role == UserRole.DRIVER.value else None
        )
        return
    
    # Use the appropriate agent based on the user's role
    db = get_db()
    if user.role == UserRole.DRIVER.value:
        agent = DriverAgent(db, user.id)
    elif user.role == UserRole.MANAGER.value:
        agent = ManagerAgent(db, user.id)
    elif user.role == UserRole.SHIPPER.value:
        agent = ShipperAgent(db, user.id)
    elif user.role == UserRole.CONSIGNEE.value:
        agent = ConsigneeAgent(db, user.id)
    else:
        await update.message.reply_text("Please use /start to set your role first.")
        return
    
    try:
        response = agent.process_message(text)
        await update.message.reply_text(response)
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await update.message.reply_text(
            "Sorry, I encountered an error processing your message. Please try again later."
        )

async def handle_location(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle location shared by the user."""
    user = await get_or_create_user(update.effective_user)
    location = update.message.location
    
    if user.role == UserRole.DRIVER.value:
        db = get_db()
        trip = db.query(Trip).filter(
            Trip.driver_id == user.id,
            Trip.status != TripStatus.COMPLETED.value
        ).order_by(Trip.created_at.desc()).first()
        
        if trip:
            # Save location
            new_location = Location(
                trip_id=trip.id,
                latitude=location.latitude,
                longitude=location.longitude
            )
            db.add(new_location)
            db.commit()
            
            await update.message.reply_text(
                "Location received and saved. Thank you!",
                reply_markup=driver_keyboard
            )
            
            # Notify stakeholders
            await notify_stakeholders(
                trip.id, 
                f"Driver location updated for Trip #{trip.id}: {location.latitude}, {location.longitude}"
            )
        else:
            await update.message.reply_text(
                "You don't have any active trips to update location for.",
                reply_markup=driver_keyboard
            )
    else:
        await update.message.reply_text("Location received, but you're not registered as a driver.")

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle callback queries from inline keyboards."""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data.startswith("set_role_"):
        role = data.replace("set_role_", "")
        user = await get_or_create_user(update.effective_user, role)
        
        # Update the user's role in the database
        db = get_db()
        db_user = db.query(User).filter(User.telegram_id == str(update.effective_user.id)).first()
        if db_user:
            db_user.role = role
            db.commit()
        
        context.user_data["role_set"] = True
        
        await query.edit_message_text(
            f"Your role has been set to {role}. You can now use the bot."
        )
        
        # Send a follow-up message with the appropriate keyboard
        if role == UserRole.DRIVER.value:
            await context.bot.send_message(
                chat_id=update.effective_user.id,
                text="As a driver, you can use these quick reply buttons:",
                reply_markup=driver_keyboard
            )
        else:
            await context.bot.send_message(
                chat_id=update.effective_user.id,
                text=f"As a {role}, you can ask me questions about your shipments and I'll help you manage them."
            )
    elif data.startswith("confirm_trip_"):
        trip_id = int(data.split("_")[-1])
        await confirm_trip(update, context, trip_id)
    elif data.startswith("survey_"):
        parts = data.split("_")
        rating = parts[1]
        trip_id = int(parts[2])
        
        await query.edit_message_text(f"Thank you for your feedback! You rated this trip as: {rating}")
        
        # Here you could store the survey response in the database

async def update_trip_status(update: Update, context: ContextTypes.DEFAULT_TYPE, trip_id: int, status: str, notes: str = None) -> None:
    """Update the status of a trip."""
    user = await get_or_create_user(update.effective_user)
    db = get_db()
    
    # Update trip status
    trip = db.query(Trip).filter(Trip.id == trip_id).first()
    if trip:
        trip.status = status
        trip.updated_at = datetime.utcnow()
        
        # Create status update
        status_update = StatusUpdate(
            trip_id=trip_id,
            user_id=user.id,
            status=status,
            notes=notes
        )
        db.add(status_update)
        db.commit()
        
        await update.message.reply_text(f"Status updated to: {status}")
    else:
        await update.message.reply_text(f"Trip #{trip_id} not found.")

async def update_message(update: Update, context: ContextTypes.DEFAULT_TYPE, message: str) -> None:
    """Send a simple update message."""
    await update.message.reply_text(message)

async def confirm_trip(update: Update, context: ContextTypes.DEFAULT_TYPE, trip_id: int) -> None:
    """Confirm a trip assignment."""
    user = await get_or_create_user(update.effective_user)
    db = get_db()
    
    trip = db.query(Trip).filter(Trip.id == trip_id).first()
    if trip and trip.driver_id == user.id:
        # Create status update
        status_update = StatusUpdate(
            trip_id=trip_id,
            user_id=user.id,
            status=TripStatus.ASSIGNED.value,
            notes="Trip confirmed by driver"
        )
        db.add(status_update)
        db.commit()
        
        await update.callback_query.edit_message_text(
            f"Trip #{trip_id} confirmed. You will receive updates and can use the quick reply buttons to report your status."
        )
        
        # Notify stakeholders
        await notify_stakeholders(trip_id, f"Driver has confirmed Trip #{trip_id}")
    else:
        await update.callback_query.edit_message_text(f"Trip #{trip_id} not found or not assigned to you.")

def create_application():
    """Create the Application and add handlers."""
    # Create the Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("set_role", set_role_command))
    
    # Message handlers
    application.add_handler(MessageHandler(filters.LOCATION, handle_location))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    
    # Callback query handler
    application.add_handler(CallbackQueryHandler(handle_callback_query))

    return application 