import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
from sqlalchemy.orm import Session

from app.models import User, Trip, StatusUpdate, Location, Issue, Notification
from app.schemas import TripStatus, UserRole

def format_datetime(dt: datetime) -> str:
    """Format a datetime object to a readable string."""
    if not dt:
        return "Not specified"
    return dt.strftime("%Y-%m-%d %H:%M")

def format_trip_details(trip: Trip) -> str:
    """Format trip details for display in messages."""
    details = f"""
    Trip #{trip.id}:
    
    Status: {trip.status}
    Pickup: {trip.pickup_address}
    Delivery: {trip.delivery_address}
    Cargo: {trip.cargo_description}
    """
    
    if trip.pickup_time_window_start and trip.pickup_time_window_end:
        details += f"Pickup window: {format_datetime(trip.pickup_time_window_start)} to {format_datetime(trip.pickup_time_window_end)}\n"
    
    if trip.delivery_time_window_start and trip.delivery_time_window_end:
        details += f"Delivery window: {format_datetime(trip.delivery_time_window_start)} to {format_datetime(trip.delivery_time_window_end)}\n"
    
    return details

def get_active_trip_for_user(db: Session, user_id: int) -> Optional[Trip]:
    """Get the active trip for a user."""
    return db.query(Trip).filter(
        Trip.driver_id == user_id,
        Trip.status != TripStatus.COMPLETED.value
    ).order_by(Trip.created_at.desc()).first()

def get_stakeholders_for_trip(db: Session, trip_id: int) -> List[User]:
    """Get all stakeholders for a trip."""
    trip = db.query(Trip).filter(Trip.id == trip_id).first()
    if not trip:
        return []
    
    stakeholders = []
    for role_id, role_name in [
        (trip.manager_id, UserRole.MANAGER.value),
        (trip.shipper_id, UserRole.SHIPPER.value),
        (trip.consignee_id, UserRole.CONSIGNEE.value)
    ]:
        if role_id:
            user = db.query(User).filter(User.id == role_id).first()
            if user:
                stakeholders.append(user)
    
    return stakeholders

def create_notification_for_stakeholders(
    db: Session, 
    trip_id: int, 
    message: str,
    exclude_user_id: Optional[int] = None
) -> List[Notification]:
    """Create notifications for all stakeholders of a trip."""
    stakeholders = get_stakeholders_for_trip(db, trip_id)
    notifications = []
    
    for stakeholder in stakeholders:
        if exclude_user_id and stakeholder.id == exclude_user_id:
            continue
            
        notification = Notification(
            user_id=stakeholder.id,
            trip_id=trip_id,
            message=message
        )
        db.add(notification)
        notifications.append(notification)
    
    db.commit()
    return notifications

def calculate_eta(
    trip: Trip, 
    current_location: Location, 
    average_speed_kmh: float = 60.0
) -> Optional[datetime]:
    """
    Calculate estimated time of arrival based on current location.
    This is a simplified calculation and would need to be replaced with
    a proper routing service in a production environment.
    """
    if not trip.delivery_lat or not trip.delivery_lng:
        return None
        
    # Calculate distance (very simplified, doesn't account for roads)
    from math import radians, cos, sin, asin, sqrt
    
    def haversine(lon1, lat1, lon2, lat2):
        # Convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r
    
    distance_km = haversine(
        current_location.longitude, 
        current_location.latitude, 
        trip.delivery_lng, 
        trip.delivery_lat
    )
    
    # Calculate time
    hours_needed = distance_km / average_speed_kmh
    eta = current_location.timestamp + timedelta(hours=hours_needed)
    
    return eta 