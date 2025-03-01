from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from datetime import datetime

from app.models import User, Trip, StatusUpdate, Location, Issue, Notification
from app.schemas import TripStatus, UserRole
from app.utils import format_trip_details, calculate_eta

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3-70b-8192")

# Initialize the LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=LLM_MODEL
)

class ShipperAgent:
    def __init__(self, db: Session, shipper_id: int):
        self.db = db
        self.shipper_id = shipper_id
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent_executor = self._create_agent()

    def _create_agent(self) -> AgentExecutor:
        tools = [
            Tool(
                name="get_all_shipments",
                func=self._get_all_shipments,
                description="Get all shipments for this shipper"
            ),
            Tool(
                name="get_shipment_details",
                func=self._get_shipment_details,
                description="Get detailed information about a specific shipment"
            ),
            Tool(
                name="get_shipment_status",
                func=self._get_shipment_status,
                description="Get the current status of a specific shipment"
            ),
            Tool(
                name="get_driver_location",
                func=self._get_driver_location,
                description="Get the current location of the driver for a specific shipment"
            ),
            Tool(
                name="get_shipment_issues",
                func=self._get_shipment_issues,
                description="Get all issues reported for a specific shipment"
            ),
            Tool(
                name="send_message_to_driver",
                func=self._send_message_to_driver,
                description="Send a message to the driver of a specific shipment"
            ),
            Tool(
                name="send_message_to_manager",
                func=self._send_message_to_manager,
                description="Send a message to the manager of a specific shipment"
            )
        ]

        prompt = PromptTemplate.from_template(
            """You are an AI assistant for shippers using a logistics management system.
            Your job is to help shippers track their shipments, communicate with drivers and managers, and resolve issues.
            
            Chat History:
            {chat_history}
            
            Human: {input}
            
            Think about how to best help the shipper. You have access to the following tools:
            
            {tools}
            
            Use the tools to provide accurate and helpful responses. If you don't know something, say so.
            
            AI: """
        )

        agent = create_react_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, memory=self.memory, verbose=True)

    def process_message(self, message: str) -> str:
        """Process a message from the shipper and return a response"""
        response = self.agent_executor.invoke({"input": message})
        return response["output"]

    def _get_all_shipments(self) -> str:
        """Get all shipments for this shipper"""
        trips = self.db.query(Trip).filter(
            Trip.shipper_id == self.shipper_id
        ).order_by(Trip.created_at.desc()).all()
        
        if not trips:
            return "You don't have any shipments at the moment."
        
        response = "Your shipments:\n\n"
        for trip in trips:
            response += f"""
            Shipment #{trip.id}:
            - Status: {trip.status}
            - Pickup: {trip.pickup_address}
            - Delivery: {trip.delivery_address}
            - Cargo: {trip.cargo_description}
            """
        
        return response

    def _get_shipment_details(self, trip_id: int) -> str:
        """Get detailed information about a specific shipment"""
        trip = self.db.query(Trip).filter(
            Trip.id == trip_id,
            Trip.shipper_id == self.shipper_id
        ).first()
        
        if not trip:
            return f"Shipment #{trip_id} not found or you don't have access to it."
        
        # Get the driver information
        driver_info = ""
        if trip.driver:
            driver_info = f"""
            Driver: {trip.driver.first_name} {trip.driver.last_name}
            Phone: {trip.driver.phone_number}
            """
        else:
            driver_info = "No driver assigned yet."
        
        # Get the manager information
        manager_info = ""
        if trip.manager:
            manager_info = f"""
            Manager: {trip.manager.first_name} {trip.manager.last_name}
            Phone: {trip.manager.phone_number}
            """
        else:
            manager_info = "No manager assigned."
        
        # Get the consignee information
        consignee_info = ""
        if trip.consignee:
            consignee_info = f"""
            Consignee: {trip.consignee.first_name} {trip.consignee.last_name}
            Phone: {trip.consignee.phone_number}
            """
        else:
            consignee_info = "No consignee assigned."
        
        response = f"""
        Shipment #{trip.id} Details:
        
        Status: {trip.status}
        
        {driver_info}
        
        {manager_info}
        
        {consignee_info}
        
        Pickup: {trip.pickup_address}
        """
        
        if trip.pickup_time_window_start and trip.pickup_time_window_end:
            response += f"Pickup window: {trip.pickup_time_window_start.strftime('%Y-%m-%d %H:%M')} to {trip.pickup_time_window_end.strftime('%Y-%m-%d %H:%M')}\n"
        
        response += f"""
        Delivery: {trip.delivery_address}
        """
        
        if trip.delivery_time_window_start and trip.delivery_time_window_end:
            response += f"Delivery window: {trip.delivery_time_window_start.strftime('%Y-%m-%d %H:%M')} to {trip.delivery_time_window_end.strftime('%Y-%m-%d %H:%M')}\n"
        
        response += f"""
        Cargo: {trip.cargo_description}
        Weight: {trip.cargo_weight} kg
        Volume: {trip.cargo_volume} m³
        """
        
        return response

    def _get_shipment_status(self, trip_id: int) -> str:
        """Get the current status of a specific shipment"""
        trip = self.db.query(Trip).filter(
            Trip.id == trip_id,
            Trip.shipper_id == self.shipper_id
        ).first()
        
        if not trip:
            return f"Shipment #{trip_id} not found or you don't have access to it."
        
        # Get the latest status update
        latest_update = self.db.query(StatusUpdate).filter(
            StatusUpdate.trip_id == trip_id
        ).order_by(StatusUpdate.created_at.desc()).first()
        
        response = f"""
        Shipment #{trip.id} Status:
        
        Current status: {trip.status}
        """
        
        if latest_update:
            user = self.db.query(User).filter(User.id == latest_update.user_id).first()
            user_name = f"{user.first_name} {user.last_name}" if user else "Unknown user"
            
            response += f"""
            Last update: {latest_update.created_at.strftime('%Y-%m-%d %H:%M')}
            Updated by: {user_name}
            """
            
            if latest_update.notes:
                response += f"Notes: {latest_update.notes}\n"
        
        # Get the latest location and calculate ETA
        latest_location = self.db.query(Location).filter(
            Location.trip_id == trip_id
        ).order_by(Location.timestamp.desc()).first()
        
        if latest_location:
            response += f"""
            Last known location: {latest_location.latitude}, {latest_location.longitude}
            Location time: {latest_location.timestamp.strftime('%Y-%m-%d %H:%M')}
            """
            
            # Calculate ETA if possible
            if trip.delivery_lat and trip.delivery_lng:
                eta = calculate_eta(trip, latest_location)
                if eta:
                    response += f"Estimated arrival: {eta.strftime('%Y-%m-%d %H:%M')}\n"
        
        return response

    def _get_driver_location(self, trip_id: int) -> str:
        """Get the current location of the driver for a specific shipment"""
        trip = self.db.query(Trip).filter(
            Trip.id == trip_id,
            Trip.shipper_id == self.shipper_id
        ).first()
        
        if not trip:
            return f"Shipment #{trip_id} not found or you don't have access to it."
        
        if not trip.driver_id:
            return f"Shipment #{trip_id} does not have a driver assigned."
        
        # Get the latest location
        latest_location = self.db.query(Location).filter(
            Location.trip_id == trip_id
        ).order_by(Location.timestamp.desc()).first()
        
        if not latest_location:
            return f"No location data available for Shipment #{trip_id}."
        
        response = f"""
        Driver location for Shipment #{trip_id}:
        
        Latitude: {latest_location.latitude}
        Longitude: {latest_location.longitude}
        Time: {latest_location.timestamp.strftime('%Y-%m-%d %H:%M')}
        """
        
        # Calculate ETA if possible
        if trip.delivery_lat and trip.delivery_lng:
            eta = calculate_eta(trip, latest_location)
            if eta:
                response += f"Estimated arrival: {eta.strftime('%Y-%m-%d %H:%M')}\n"
        
        return response

    def _get_shipment_issues(self, trip_id: int) -> str:
        """Get all issues reported for a specific shipment"""
        trip = self.db.query(Trip).filter(
            Trip.id == trip_id,
            Trip.shipper_id == self.shipper_id
        ).first()
        
        if not trip:
            return f"Shipment #{trip_id} not found or you don't have access to it."
        
        issues = self.db.query(Issue).filter(
            Issue.trip_id == trip_id
        ).order_by(Issue.created_at.desc()).all()
        
        if not issues:
            return f"No issues reported for Shipment #{trip_id}."
        
        response = f"Issues for Shipment #{trip_id}:\n\n"
        for issue in issues:
            user = self.db.query(User).filter(User.id == issue.reported_by_id).first()
            user_name = f"{user.first_name} {user.last_name}" if user else "Unknown user"
            
            response += f"""
            Issue #{issue.id}:
            Status: {issue.status}
            Reported by: {user_name}
            Time: {issue.created_at.strftime('%Y-%m-%d %H:%M')}
            Description: {issue.description}
            """
            
            if issue.resolved_at:
                response += f"Resolved at: {issue.resolved_at.strftime('%Y-%m-%d %H:%M')}\n"
        
        return response

    def _send_message_to_driver(self, trip_id: int, message: str) -> str:
        """Send a message to the driver of a specific shipment"""
        trip = self.db.query(Trip).filter(
            Trip.id == trip_id,
            Trip.shipper_id == self.shipper_id
        ).first()
        
        if not trip:
            return f"Shipment #{trip_id} not found or you don't have access to it."
        
        if not trip.driver_id:
            return f"Shipment #{trip_id} does not have a driver assigned."
        
        notification = Notification(
            user_id=trip.driver_id,
            trip_id=trip.id,
            message=f"Message from shipper: {message}"
        )
        self.db.add(notification)
        self.db.commit()
        
        return f"Message sent to the driver of Shipment #{trip_id}."

    def _send_message_to_manager(self, trip_id: int, message: str) -> str:
        """Send a message to the manager of a specific shipment"""
        trip = self.db.query(Trip).filter(
            Trip.id == trip_id,
            Trip.shipper_id == self.shipper_id
        ).first()
        
        if not trip:
            return f"Shipment #{trip_id} not found or you don't have access to it."
        
        if not trip.manager_id:
            return f"Shipment #{trip_id} does not have a manager assigned."
        
        notification = Notification(
            user_id=trip.manager_id,
            trip_id=trip.id,
            message=f"Message from shipper: {message}"
        )
        self.db.add(notification)
        self.db.commit()
        
        return f"Message sent to the manager of Shipment #{trip_id}." 