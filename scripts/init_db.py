import sys
import os
from datetime import datetime, timedelta
import random

# Add the parent directory to the path so we can import the app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal, engine, Base
from app.models import User, Trip, StatusUpdate, Location, Issue, Notification
from app.schemas import UserRole, TripStatus

# Create tables
Base.metadata.create_all(bind=engine)

# Create a database session
db = SessionLocal()

def create_test_data():
    """Create test data for the application."""
    print("Creating test data...")
    
    # Create users
    driver = User(
        telegram_id="123456789",
        username="test_driver",
        first_name="Test",
        last_name="Driver",
        phone_number="+1234567890",
        role=UserRole.DRIVER.value
    )
    
    manager = User(
        telegram_id="987654321",
        username="test_manager",
        first_name="Test",
        last_name="Manager",
        phone_number="+9876543210",
        role=UserRole.MANAGER.value
    )
    
    shipper = User(
        telegram_id="123123123",
        username="test_shipper",
        first_name="Test",
        last_name="Shipper",
        phone_number="+1231231230",
        role=UserRole.SHIPPER.value
    )
    
    consignee = User(
        telegram_id="456456456",
        username="test_consignee",
        first_name="Test",
        last_name="Consignee",
        phone_number="+4564564560",
        role=UserRole.CONSIGNEE.value
    )
    
    db.add_all([driver, manager, shipper, consignee])
    db.commit()
    
    # Create a trip
    now = datetime.utcnow()
    trip = Trip(
        driver_id=driver.id,
        manager_id=manager.id,
        shipper_id=shipper.id,
        consignee_id=consignee.id,
        pickup_address="123 Pickup St, Pickup City, PC 12345",
        pickup_lat=37.7749,
        pickup_lng=-122.4194,
        pickup_time_window_start=now + timedelta(hours=1),
        pickup_time_window_end=now + timedelta(hours=3),
        delivery_address="456 Delivery Ave, Delivery City, DC 67890",
        delivery_lat=34.0522,
        delivery_lng=-118.2437,
        delivery_time_window_start=now + timedelta(days=1),
        delivery_time_window_end=now + timedelta(days=1, hours=2),
        cargo_description="10 pallets of electronics",
        cargo_weight=2000.0,
        cargo_volume=15.0,
        status=TripStatus.ASSIGNED.value
    )
    
    db.add(trip)
    db.commit()
    
    # Create status updates
    status_update = StatusUpdate(
        trip_id=trip.id,
        user_id=driver.id,
        status=TripStatus.ASSIGNED.value,
        notes="Trip assigned to driver"
    )
    
    db.add(status_update)
    db.commit()
    
    # Create locations (simulating a route)
    locations = []
    start_lat, start_lng = trip.pickup_lat, trip.pickup_lng
    end_lat, end_lng = trip.delivery_lat, trip.delivery_lng
    
    # Create 10 points along the route
    for i in range(10):
        factor = i / 10.0
        lat = start_lat + (end_lat - start_lat) * factor
        lng = start_lng + (end_lng - start_lng) * factor
        
        # Add some randomness to make it look more realistic
        lat += random.uniform(-0.01, 0.01)
        lng += random.uniform(-0.01, 0.01)
        
        location = Location(
            trip_id=trip.id,
            latitude=lat,
            longitude=lng,
            timestamp=now + timedelta(hours=i)
        )
        locations.append(location)
    
    db.add_all(locations)
    db.commit()
    
    # Create an issue
    issue = Issue(
        trip_id=trip.id,
        reported_by_id=driver.id,
        description="Traffic delay on highway",
        status="open"
    )
    
    db.add(issue)
    db.commit()
    
    # Create notifications
    notification_manager = Notification(
        user_id=manager.id,
        trip_id=trip.id,
        message=f"Driver has been assigned to Trip #{trip.id}"
    )
    
    notification_shipper = Notification(
        user_id=shipper.id,
        trip_id=trip.id,
        message=f"Driver has been assigned to Trip #{trip.id}"
    )
    
    notification_consignee = Notification(
        user_id=consignee.id,
        trip_id=trip.id,
        message=f"Driver has been assigned to Trip #{trip.id}"
    )
    
    db.add_all([notification_manager, notification_shipper, notification_consignee])
    db.commit()
    
    print("Test data created successfully!")
    print(f"Driver ID: {driver.id}, Telegram ID: {driver.telegram_id}")
    print(f"Manager ID: {manager.id}, Telegram ID: {manager.telegram_id}")
    print(f"Shipper ID: {shipper.id}, Telegram ID: {shipper.telegram_id}")
    print(f"Consignee ID: {consignee.id}, Telegram ID: {consignee.telegram_id}")
    print(f"Trip ID: {trip.id}")

if __name__ == "__main__":
    create_test_data() 