import os
from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import uvicorn
from dotenv import load_dotenv
import json
from telegram import Update
from telegram.ext import Application

from app.database import engine, get_db, Base
from app.models import User, Trip, StatusUpdate, Location, Issue, Notification
from app.schemas import (
    UserCreate, User as UserSchema,
    TripCreate, Trip as TripSchema, TripUpdate,
    StatusUpdateCreate, StatusUpdate as StatusUpdateSchema,
    LocationCreate, Location as LocationSchema,
    IssueCreate, Issue as IssueSchema,
    NotificationCreate, Notification as NotificationSchema,
    UserRole, TripStatus
)
from app.bot import create_application
from app.agents import DriverAgent, ManagerAgent, ShipperAgent, ConsigneeAgent

# Load environment variables
load_dotenv()
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Create database tables
Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI(title="Logistik AI API")

# Create Telegram bot application
bot_app = create_application()

@app.on_event("startup")
async def startup_event():
    """Set up the Telegram bot webhook on startup."""
    if WEBHOOK_URL and TELEGRAM_BOT_TOKEN:
        webhook_url = f"{WEBHOOK_URL}/webhook/{TELEGRAM_BOT_TOKEN}"
        # Initialize the bot application
        await bot_app.initialize()
        await bot_app.bot.set_webhook(url=webhook_url)
        print(f"Webhook set to {webhook_url}")
    else:
        print("WEBHOOK_URL or TELEGRAM_BOT_TOKEN not set. Webhook not configured.")

@app.post(f"/webhook/{TELEGRAM_BOT_TOKEN}")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle Telegram webhook updates."""
    data = await request.json()
    background_tasks.add_task(process_update, data)
    return JSONResponse(content={"status": "ok"})

async def process_update(data: dict):
    """Process Telegram update in the background."""
    update = Update.de_json(data=data, bot=bot_app.bot)
    # Ensure the bot application is initialized
    try:
        # This will raise an exception if not initialized
        bot_app._check_initialized()
    except RuntimeError:
        # Initialize if not already initialized
        await bot_app.initialize()
    await bot_app.process_update(update)

# API endpoints for Users
@app.post("/users/", response_model=UserSchema)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(
        telegram_id=user.telegram_id,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name,
        phone_number=user.phone_number,
        role=user.role.value
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users/{user_id}", response_model=UserSchema)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

# API endpoints for Trips
@app.post("/trips/", response_model=TripSchema)
def create_trip(trip: TripCreate, db: Session = Depends(get_db)):
    db_trip = Trip(
        driver_id=trip.driver_id,
        shipper_id=trip.shipper_id,
        consignee_id=trip.consignee_id,
        manager_id=trip.manager_id,
        pickup_address=trip.pickup_address,
        pickup_lat=trip.pickup_lat,
        pickup_lng=trip.pickup_lng,
        pickup_time_window_start=trip.pickup_time_window_start,
        pickup_time_window_end=trip.pickup_time_window_end,
        delivery_address=trip.delivery_address,
        delivery_lat=trip.delivery_lat,
        delivery_lng=trip.delivery_lng,
        delivery_time_window_start=trip.delivery_time_window_start,
        delivery_time_window_end=trip.delivery_time_window_end,
        cargo_description=trip.cargo_description,
        cargo_weight=trip.cargo_weight,
        cargo_volume=trip.cargo_volume,
        status=TripStatus.ASSIGNED.value
    )
    db.add(db_trip)
    db.commit()
    db.refresh(db_trip)
    
    # If a driver is assigned, send them a notification via Telegram
    if db_trip.driver_id:
        driver = db.query(User).filter(User.id == db_trip.driver_id).first()
        if driver and driver.telegram_id:
            # Create a message with trip details
            message = f"""
            🚚 New Trip Assignment #{db_trip.id}:
            
            📍 Pickup: {db_trip.pickup_address}
            🏁 Delivery: {db_trip.delivery_address}
            📦 Cargo: {db_trip.cargo_description}
            """
            
            if db_trip.pickup_time_window_start and db_trip.pickup_time_window_end:
                message += f"⏰ Pickup window: {db_trip.pickup_time_window_start.strftime('%Y-%m-%d %H:%M')} to {db_trip.pickup_time_window_end.strftime('%Y-%m-%d %H:%M')}\n"
            
            if db_trip.delivery_time_window_start and db_trip.delivery_time_window_end:
                message += f"⏰ Delivery window: {db_trip.delivery_time_window_start.strftime('%Y-%m-%d %H:%M')} to {db_trip.delivery_time_window_end.strftime('%Y-%m-%d %H:%M')}\n"
            
            # Create inline keyboard for confirmation
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("Confirm Trip", callback_data=f"confirm_trip_{db_trip.id}")]
            ])
            
            # Send message to driver
            bot_app.bot.send_message(
                chat_id=driver.telegram_id,
                text=message,
                reply_markup=keyboard
            )
            
            # Send location pins
            if db_trip.pickup_lat and db_trip.pickup_lng:
                bot_app.bot.send_location(
                    chat_id=driver.telegram_id,
                    latitude=db_trip.pickup_lat,
                    longitude=db_trip.pickup_lng,
                    disable_notification=True
                )
            
            if db_trip.delivery_lat and db_trip.delivery_lng:
                bot_app.bot.send_location(
                    chat_id=driver.telegram_id,
                    latitude=db_trip.delivery_lat,
                    longitude=db_trip.delivery_lng,
                    disable_notification=True
                )
    
    return db_trip

@app.get("/trips/{trip_id}", response_model=TripSchema)
def read_trip(trip_id: int, db: Session = Depends(get_db)):
    db_trip = db.query(Trip).filter(Trip.id == trip_id).first()
    if db_trip is None:
        raise HTTPException(status_code=404, detail="Trip not found")
    return db_trip

@app.put("/trips/{trip_id}", response_model=TripSchema)
def update_trip(trip_id: int, trip_update: TripUpdate, db: Session = Depends(get_db)):
    db_trip = db.query(Trip).filter(Trip.id == trip_id).first()
    if db_trip is None:
        raise HTTPException(status_code=404, detail="Trip not found")
    
    # Update trip fields
    for key, value in trip_update.dict(exclude_unset=True).items():
        setattr(db_trip, key, value.value if key == "status" and value else value)
    
    db.commit()
    db.refresh(db_trip)
    return db_trip

# API endpoints for Status Updates
@app.post("/status-updates/", response_model=StatusUpdateSchema)
def create_status_update(status_update: StatusUpdateCreate, db: Session = Depends(get_db)):
    db_status_update = StatusUpdate(
        trip_id=status_update.trip_id,
        user_id=status_update.user_id,
        status=status_update.status.value,
        notes=status_update.notes
    )
    db.add(db_status_update)
    
    # Update trip status
    trip = db.query(Trip).filter(Trip.id == status_update.trip_id).first()
    if trip:
        trip.status = status_update.status.value
    
    db.commit()
    db.refresh(db_status_update)
    return db_status_update

# API endpoints for Locations
@app.post("/locations/", response_model=LocationSchema)
def create_location(location: LocationCreate, db: Session = Depends(get_db)):
    db_location = Location(
        trip_id=location.trip_id,
        latitude=location.latitude,
        longitude=location.longitude
    )
    db.add(db_location)
    db.commit()
    db.refresh(db_location)
    return db_location

# API endpoints for Issues
@app.post("/issues/", response_model=IssueSchema)
def create_issue(issue: IssueCreate, db: Session = Depends(get_db)):
    db_issue = Issue(
        trip_id=issue.trip_id,
        reported_by_id=issue.reported_by_id,
        description=issue.description,
        status="open"
    )
    db.add(db_issue)
    
    # Update trip status
    trip = db.query(Trip).filter(Trip.id == issue.trip_id).first()
    if trip:
        trip.status = TripStatus.ISSUE_REPORTED.value
    
    db.commit()
    db.refresh(db_issue)
    return db_issue

# API endpoints for Notifications
@app.post("/notifications/", response_model=NotificationSchema)
def create_notification(notification: NotificationCreate, db: Session = Depends(get_db)):
    db_notification = Notification(
        user_id=notification.user_id,
        trip_id=notification.trip_id,
        message=notification.message
    )
    db.add(db_notification)
    db.commit()
    db.refresh(db_notification)
    
    # Send notification via Telegram if possible
    user = db.query(User).filter(User.id == notification.user_id).first()
    if user and user.telegram_id:
        try:
            bot_app.bot.send_message(
                chat_id=user.telegram_id,
                text=notification.message
            )
        except Exception as e:
            print(f"Error sending Telegram notification: {e}")
    
    return db_notification

# API endpoints for AI Agent interactions
@app.post("/agent/driver/{driver_id}/query")
def query_driver_agent(driver_id: int, query: str, db: Session = Depends(get_db)):
    """Query the driver agent with a message."""
    driver = db.query(User).filter(User.id == driver_id, User.role == UserRole.DRIVER.value).first()
    if not driver:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    agent = DriverAgent(db, driver_id)
    response = agent.process_message(query)
    
    return {"response": response}

@app.post("/agent/manager/{manager_id}/query")
def query_manager_agent(manager_id: int, query: str, db: Session = Depends(get_db)):
    """Query the manager agent with a message."""
    manager = db.query(User).filter(User.id == manager_id, User.role == UserRole.MANAGER.value).first()
    if not manager:
        raise HTTPException(status_code=404, detail="Manager not found")
    
    agent = ManagerAgent(db, manager_id)
    response = agent.process_message(query)
    
    return {"response": response}

@app.post("/agent/shipper/{shipper_id}/query")
def query_shipper_agent(shipper_id: int, query: str, db: Session = Depends(get_db)):
    """Query the shipper agent with a message."""
    shipper = db.query(User).filter(User.id == shipper_id, User.role == UserRole.SHIPPER.value).first()
    if not shipper:
        raise HTTPException(status_code=404, detail="Shipper not found")
    
    agent = ShipperAgent(db, shipper_id)
    response = agent.process_message(query)
    
    return {"response": response}

@app.post("/agent/consignee/{consignee_id}/query")
def query_consignee_agent(consignee_id: int, query: str, db: Session = Depends(get_db)):
    """Query the consignee agent with a message."""
    consignee = db.query(User).filter(User.id == consignee_id, User.role == UserRole.CONSIGNEE.value).first()
    if not consignee:
        raise HTTPException(status_code=404, detail="Consignee not found")
    
    agent = ConsigneeAgent(db, consignee_id)
    response = agent.process_message(query)
    
    return {"response": response}

@app.get("/")
def read_root():
    return {"message": "Welcome to Logistik AI API"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 