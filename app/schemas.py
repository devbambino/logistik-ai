from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

# Enums
class UserRole(str, Enum):
    DRIVER = "driver"
    MANAGER = "manager"
    SHIPPER = "shipper"
    CONSIGNEE = "consignee"

class TripStatus(str, Enum):
    ASSIGNED = "assigned"
    AT_PICKUP = "at_pickup"
    LOADING = "loading"
    IN_TRANSIT = "in_transit"
    DELAYED = "delayed"
    ISSUE_REPORTED = "issue_reported"
    AT_DESTINATION = "at_destination"
    UNLOADING = "unloading"
    COMPLETED = "completed"

class IssueStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"

# User schemas
class UserBase(BaseModel):
    telegram_id: str
    username: Optional[str] = None
    first_name: str
    last_name: Optional[str] = None
    phone_number: Optional[str] = None
    role: UserRole

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

# Trip schemas
class TripBase(BaseModel):
    pickup_address: str
    pickup_lat: Optional[float] = None
    pickup_lng: Optional[float] = None
    pickup_time_window_start: Optional[datetime] = None
    pickup_time_window_end: Optional[datetime] = None
    
    delivery_address: str
    delivery_lat: Optional[float] = None
    delivery_lng: Optional[float] = None
    delivery_time_window_start: Optional[datetime] = None
    delivery_time_window_end: Optional[datetime] = None
    
    cargo_description: str
    cargo_weight: Optional[float] = None
    cargo_volume: Optional[float] = None

class TripCreate(TripBase):
    driver_id: Optional[int] = None
    shipper_id: Optional[int] = None
    consignee_id: Optional[int] = None
    manager_id: Optional[int] = None

class TripUpdate(BaseModel):
    status: Optional[TripStatus] = None
    driver_id: Optional[int] = None
    shipper_id: Optional[int] = None
    consignee_id: Optional[int] = None
    manager_id: Optional[int] = None

class Trip(TripBase):
    id: int
    driver_id: Optional[int] = None
    shipper_id: Optional[int] = None
    consignee_id: Optional[int] = None
    manager_id: Optional[int] = None
    status: TripStatus
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

# Status update schemas
class StatusUpdateBase(BaseModel):
    trip_id: int
    user_id: int
    status: TripStatus
    notes: Optional[str] = None

class StatusUpdateCreate(StatusUpdateBase):
    pass

class StatusUpdate(StatusUpdateBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True

# Location schemas
class LocationBase(BaseModel):
    trip_id: int
    latitude: float
    longitude: float

class LocationCreate(LocationBase):
    pass

class Location(LocationBase):
    id: int
    timestamp: datetime

    class Config:
        orm_mode = True

# Issue schemas
class IssueBase(BaseModel):
    trip_id: int
    reported_by_id: int
    description: str

class IssueCreate(IssueBase):
    pass

class IssueUpdate(BaseModel):
    status: Optional[IssueStatus] = None
    resolved_at: Optional[datetime] = None

class Issue(IssueBase):
    id: int
    status: IssueStatus
    resolved_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

# Notification schemas
class NotificationBase(BaseModel):
    user_id: int
    trip_id: Optional[int] = None
    message: str

class NotificationCreate(NotificationBase):
    pass

class Notification(NotificationBase):
    id: int
    is_read: bool
    created_at: datetime

    class Config:
        orm_mode = True 