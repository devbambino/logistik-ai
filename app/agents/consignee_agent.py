from langchain.agents import Tool, AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

class ConsigneeAgent:
    def __init__(self, db: Session, consignee_id: int):
        self.db = db
        self.consignee_id = consignee_id
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent_executor = self._create_agent()

    def _create_agent(self) -> AgentExecutor:
        tools = [
            Tool(
                name="get_all_incoming_shipments",
                func=self._get_all_incoming_shipments,
                description="Get all incoming shipments for this consignee"
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
                name="get_estimated_arrival",
                func=self._get_estimated_arrival,
                description="Get the estimated arrival time for a specific shipment"
            ),
            Tool(
                name="send_message_to_driver",
                func=self._send_message_to_driver,
                description="Send a message to the driver of a specific shipment"
            ),
            Tool(
                name="send_message_to_shipper",
                func=self._send_message_to_shipper,
                description="Send a message to the shipper of a specific shipment"
            )
        ]

        # Create a custom agent using the Groq model
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant for consignees (recipients) using a logistics management system. Your job is to help consignees track incoming shipments, communicate with drivers and shippers, and prepare for deliveries."),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent without binding tools directly
        agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x.get("chat_history", []),
                "agent_scratchpad": lambda x: format_to_openai_function_messages(x.get("intermediate_steps", [])),
            }
            | prompt
            | llm
            | OpenAIFunctionsAgentOutputParser()
        )
        
        return AgentExecutor(agent=agent, tools=tools, memory=self.memory, verbose=True)

    def process_message(self, message: str) -> str:
        """Process a message from the consignee and return a response"""
        response = self.agent_executor.invoke({"input": message})
        return response["output"]

    def _get_all_incoming_shipments(self) -> str:
        """Get all incoming shipments for this consignee"""
        trips = self.db.query(Trip).filter(
            Trip.consignee_id == self.consignee_id
        ).order_by(Trip.created_at.desc()).all()
        
        if not trips:
            return "You don't have any incoming shipments at the moment."
        
        response = "Your incoming shipments:\n\n"
        for trip in trips:
            response += f"""
            Shipment #{trip.id}:
            - Status: {trip.status}
            - From: {trip.pickup_address}
            - To: {trip.delivery_address}
            - Cargo: {trip.cargo_description}
            """
            
            if trip.delivery_time_window_start and trip.delivery_time_window_end:
                response += f"- Expected delivery: {trip.delivery_time_window_start.strftime('%Y-%m-%d %H:%M')} to {trip.delivery_time_window_end.strftime('%Y-%m-%d %H:%M')}\n"
        
        return response

    def _get_shipment_details(self, trip_id: int) -> str:
        """Get detailed information about a specific shipment"""
        trip = self.db.query(Trip).filter(
            Trip.id == trip_id,
            Trip.consignee_id == self.consignee_id
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
        
        # Get the shipper information
        shipper_info = ""
        if trip.shipper:
            shipper_info = f"""
            Shipper: {trip.shipper.first_name} {trip.shipper.last_name}
            Phone: {trip.shipper.phone_number}
            """
        else:
            shipper_info = "No shipper assigned."
        
        response = f"""
        Shipment #{trip.id} Details:
        
        Status: {trip.status}
        
        {driver_info}
        
        {shipper_info}
        
        From: {trip.pickup_address}
        To: {trip.delivery_address}
        
        Cargo: {trip.cargo_description}
        Weight: {trip.cargo_weight} kg
        Volume: {trip.cargo_volume} m³
        """
        
        if trip.delivery_time_window_start and trip.delivery_time_window_end:
            response += f"Expected delivery: {trip.delivery_time_window_start.strftime('%Y-%m-%d %H:%M')} to {trip.delivery_time_window_end.strftime('%Y-%m-%d %H:%M')}\n"
        
        return response

    def _get_shipment_status(self, trip_id: int) -> str:
        """Get the current status of a specific shipment"""
        trip = self.db.query(Trip).filter(
            Trip.id == trip_id,
            Trip.consignee_id == self.consignee_id
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
        
        return response

    def _get_driver_location(self, trip_id: int) -> str:
        """Get the current location of the driver for a specific shipment"""
        trip = self.db.query(Trip).filter(
            Trip.id == trip_id,
            Trip.consignee_id == self.consignee_id
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
        
        return response

    def _get_estimated_arrival(self, trip_id: int) -> str:
        """Get the estimated arrival time for a specific shipment"""
        trip = self.db.query(Trip).filter(
            Trip.id == trip_id,
            Trip.consignee_id == self.consignee_id
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
        
        # Calculate ETA
        if not trip.delivery_lat or not trip.delivery_lng:
            return f"Delivery coordinates not available for Shipment #{trip_id}."
        
        eta = calculate_eta(trip, latest_location)
        if not eta:
            return f"Unable to calculate ETA for Shipment #{trip_id}."
        
        response = f"""
        Estimated arrival for Shipment #{trip_id}:
        
        ETA: {eta.strftime('%Y-%m-%d %H:%M')}
        
        Current status: {trip.status}
        Last location update: {latest_location.timestamp.strftime('%Y-%m-%d %H:%M')}
        """
        
        if trip.delivery_time_window_start and trip.delivery_time_window_end:
            response += f"""
            Scheduled delivery window: {trip.delivery_time_window_start.strftime('%Y-%m-%d %H:%M')} to {trip.delivery_time_window_end.strftime('%Y-%m-%d %H:%M')}
            """
            
            # Check if the ETA is within the delivery window
            if eta < trip.delivery_time_window_start:
                response += "The driver is expected to arrive EARLY.\n"
            elif eta > trip.delivery_time_window_end:
                response += "The driver is expected to arrive LATE.\n"
            else:
                response += "The driver is expected to arrive ON TIME.\n"
        
        return response

    def _send_message_to_driver(self, trip_id: int, message: str) -> str:
        """Send a message to the driver of a specific shipment"""
        trip = self.db.query(Trip).filter(
            Trip.id == trip_id,
            Trip.consignee_id == self.consignee_id
        ).first()
        
        if not trip:
            return f"Shipment #{trip_id} not found or you don't have access to it."
        
        if not trip.driver_id:
            return f"Shipment #{trip_id} does not have a driver assigned."
        
        notification = Notification(
            user_id=trip.driver_id,
            trip_id=trip.id,
            message=f"Message from consignee: {message}"
        )
        self.db.add(notification)
        self.db.commit()
        
        return f"Message sent to the driver of Shipment #{trip_id}."

    def _send_message_to_shipper(self, trip_id: int, message: str) -> str:
        """Send a message to the shipper of a specific shipment"""
        trip = self.db.query(Trip).filter(
            Trip.id == trip_id,
            Trip.consignee_id == self.consignee_id
        ).first()
        
        if not trip:
            return f"Shipment #{trip_id} not found or you don't have access to it."
        
        if not trip.shipper_id:
            return f"Shipment #{trip_id} does not have a shipper assigned."
        
        notification = Notification(
            user_id=trip.shipper_id,
            trip_id=trip.id,
            message=f"Message from consignee: {message}"
        )
        self.db.add(notification)
        self.db.commit()
        
        return f"Message sent to the shipper of Shipment #{trip_id}." 