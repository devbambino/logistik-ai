from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Text, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum
from datetime import datetime

Base = declarative_base()

class UserRole(enum.Enum):
    DRIVER = "driver"
    MANAGER = "manager"
    SHIPPER = "shipper"
    CONSIGNEE = "consignee"

class TripStatus(enum.Enum):
    ASSIGNED = "assigned"
    AT_PICKUP = "at_pickup"
    LOADING = "loading"
    IN_TRANSIT = "in_transit"
    DELAYED = "delayed"
    ISSUE_REPORTED = "issue_reported"
    AT_DESTINATION = "at_destination"
    UNLOADING = "unloading"
    COMPLETED = "completed"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(String, unique=True, index=True)
    username = Column(String, nullable=True)
    first_name = Column(String)
    last_name = Column(String, nullable=True)
    phone_number = Column(String, nullable=True)
    role = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    trips_as_driver = relationship("Trip", back_populates="driver", foreign_keys="Trip.driver_id")
    trips_as_shipper = relationship("Trip", back_populates="shipper", foreign_keys="Trip.shipper_id")
    trips_as_consignee = relationship("Trip", back_populates="consignee", foreign_keys="Trip.consignee_id")
    trips_as_manager = relationship("Trip", back_populates="manager", foreign_keys="Trip.manager_id")
    status_updates = relationship("StatusUpdate", back_populates="user")
    issues = relationship("Issue", back_populates="reported_by")

class Trip(Base):
    __tablename__ = "trips"

    id = Column(Integer, primary_key=True, index=True)
    driver_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    shipper_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    consignee_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    manager_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Trip details
    pickup_address = Column(String, nullable=False)
    pickup_lat = Column(Float, nullable=True)
    pickup_lng = Column(Float, nullable=True)
    pickup_time_window_start = Column(DateTime, nullable=True)
    pickup_time_window_end = Column(DateTime, nullable=True)
    
    delivery_address = Column(String, nullable=False)
    delivery_lat = Column(Float, nullable=True)
    delivery_lng = Column(Float, nullable=True)
    delivery_time_window_start = Column(DateTime, nullable=True)
    delivery_time_window_end = Column(DateTime, nullable=True)
    
    cargo_description = Column(String, nullable=False)
    cargo_weight = Column(Float, nullable=True)
    cargo_volume = Column(Float, nullable=True)
    
    status = Column(String, default=TripStatus.ASSIGNED.value)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    driver = relationship("User", back_populates="trips_as_driver", foreign_keys=[driver_id])
    shipper = relationship("User", back_populates="trips_as_shipper", foreign_keys=[shipper_id])
    consignee = relationship("User", back_populates="trips_as_consignee", foreign_keys=[consignee_id])
    manager = relationship("User", back_populates="trips_as_manager", foreign_keys=[manager_id])
    status_updates = relationship("StatusUpdate", back_populates="trip")
    locations = relationship("Location", back_populates="trip")
    issues = relationship("Issue", back_populates="trip")

class StatusUpdate(Base):
    __tablename__ = "status_updates"

    id = Column(Integer, primary_key=True, index=True)
    trip_id = Column(Integer, ForeignKey("trips.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(String, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    trip = relationship("Trip", back_populates="status_updates")
    user = relationship("User", back_populates="status_updates")

class Location(Base):
    __tablename__ = "locations"

    id = Column(Integer, primary_key=True, index=True)
    trip_id = Column(Integer, ForeignKey("trips.id"), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships
    trip = relationship("Trip", back_populates="locations")

class Issue(Base):
    __tablename__ = "issues"

    id = Column(Integer, primary_key=True, index=True)
    trip_id = Column(Integer, ForeignKey("trips.id"), nullable=False)
    reported_by_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String, default="open")  # open, in_progress, resolved
    resolved_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    trip = relationship("Trip", back_populates="issues")
    reported_by = relationship("User", back_populates="issues")

class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    trip_id = Column(Integer, ForeignKey("trips.id"), nullable=True)
    message = Column(Text, nullable=False)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User")
    trip = relationship("Trip") 