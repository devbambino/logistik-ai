from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from datetime import datetime

from app.models import User, Trip, StatusUpdate, Location, Issue
from app.schemas import TripStatus

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3-70b-8192")

# Initialize the LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=LLM_MODEL
)

class DriverAgent:
    def __init__(self, db: Session, driver_id: int):
        self.db = db
        self.driver_id = driver_id
        self.memory = []  # Simple list to store conversation history
        self.tools = self._create_tools()
        print(f"DriverAgent initialized for driver {driver_id}")

    def _create_tools(self):
        """Create the tools for the agent"""
        tools = [
            Tool(
                name="get_current_trip",
                func=self._get_current_trip,
                description="Get the current active trip for the driver"
            ),
            Tool(
                name="update_trip_status",
                func=self._update_trip_status,
                description="Update the status of the current trip. Available statuses: at_pickup, loading, in_transit, delayed, issue_reported, at_destination, unloading, completed"
            ),
            Tool(
                name="report_issue",
                func=self._report_issue,
                description="Report an issue with the current trip"
            ),
            Tool(
                name="update_location",
                func=self._update_location,
                description="Update the current location of the driver"
            ),
            Tool(
                name="get_trip_history",
                func=self._get_trip_history,
                description="Get the history of trips for the driver"
            )
        ]
        print(f"Created tools: {[tool.name for tool in tools]}")
        return tools

    def process_message(self, message: str) -> str:
        """Process a message from the driver and return a response"""
        try:
            # Add user message to memory
            self.memory.append({"role": "user", "content": message})
            
            # Create the system message with tool descriptions
            tools_str = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
            system_message = f"""You are an AI assistant for truck drivers using a logistics management system.
            Your job is to help drivers manage their trips, update statuses, and report issues.
            
            You have access to the following tools:
            {tools_str}
            
            When a driver asks you to perform an action, use the appropriate tool.
            If you need to use a tool, respond with the tool name and the parameters in this format:
            
            Tool: <tool_name>
            Parameters: <parameter1>=<value1>, <parameter2>=<value2>, ...
            
            For example:
            Tool: update_trip_status
            Parameters: status=at_pickup, notes=Just arrived at the pickup location
            
            If you don't need to use a tool, just respond normally."""
            
            # Prepare messages for the LLM
            messages = [
                {"role": "system", "content": system_message}
            ]
            
            # Add conversation history (limited to last 10 messages)
            for msg in self.memory[-10:]:
                messages.append(msg)
            
            # Get response from LLM
            print(f"Sending message to LLM: {message}")
            response = llm.invoke(messages)
            response_content = response.content
            print(f"LLM response: {response_content}")
            
            # Add assistant response to memory
            self.memory.append({"role": "assistant", "content": response_content})
            
            # Check if the response contains a tool call
            if "Tool:" in response_content:
                print("Tool call detected in response")
                # Extract tool name and parameters
                lines = response_content.strip().split("\n")
                tool_line = next((line for line in lines if line.startswith("Tool:")), None)
                params_line = next((line for line in lines if line.startswith("Parameters:")), None)
                
                if tool_line and params_line:
                    tool_name = tool_line.replace("Tool:", "").strip()
                    params_str = params_line.replace("Parameters:", "").strip()
                    
                    print(f"Extracted tool: {tool_name}, params: {params_str}")
                    
                    # Parse parameters
                    params = {}
                    for param in params_str.split(","):
                        if "=" in param:
                            key, value = param.split("=", 1)
                            params[key.strip()] = value.strip()
                    
                    # Find the tool
                    tool = next((t for t in self.tools if t.name == tool_name), None)
                    
                    if tool:
                        print(f"Executing tool: {tool_name} with params: {params}")
                        # Execute the tool
                        tool_result = tool.func(**params)
                        
                        # Add tool result to memory
                        self.memory.append({"role": "system", "content": f"Tool result: {tool_result}"})
                        
                        # Return the result
                        return tool_result
                    else:
                        print(f"Tool not found: {tool_name}")
            
            # If no tool call or tool execution failed, return the original response
            return response_content
            
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return f"I'm sorry, I encountered an error while processing your message. Please try again or contact support if the issue persists."

    def _get_current_trip(self) -> str:
        """Get the current active trip for the driver"""
        trip = self.db.query(Trip).filter(
            Trip.driver_id == self.driver_id,
            Trip.status != TripStatus.COMPLETED.value
        ).order_by(Trip.created_at.desc()).first()
        
        if not trip:
            return "You don't have any active trips at the moment."
        
        response = f"""
        Trip #{trip.id}:
        - Status: {trip.status}
        - Pickup: {trip.pickup_address}
        - Delivery: {trip.delivery_address}
        - Cargo: {trip.cargo_description}
        """
        
        if trip.pickup_time_window_start and trip.pickup_time_window_end:
            response += f"- Pickup window: {trip.pickup_time_window_start.strftime('%Y-%m-%d %H:%M')} to {trip.pickup_time_window_end.strftime('%Y-%m-%d %H:%M')}\n"
        
        if trip.delivery_time_window_start and trip.delivery_time_window_end:
            response += f"- Delivery window: {trip.delivery_time_window_start.strftime('%Y-%m-%d %H:%M')} to {trip.delivery_time_window_end.strftime('%Y-%m-%d %H:%M')}\n"
        
        return response

    def _update_trip_status(self, status: str, notes: str = None) -> str:
        """Update the status of the current trip"""
        trip = self.db.query(Trip).filter(
            Trip.driver_id == self.driver_id,
            Trip.status != TripStatus.COMPLETED.value
        ).order_by(Trip.created_at.desc()).first()
        
        if not trip:
            return "You don't have any active trips to update."
        
        # Validate status
        try:
            new_status = TripStatus(status)
        except ValueError:
            return f"Invalid status: {status}. Available statuses: {', '.join([s.value for s in TripStatus])}"
        
        # Create status update
        status_update = StatusUpdate(
            trip_id=trip.id,
            user_id=self.driver_id,
            status=new_status.value,
            notes=notes
        )
        self.db.add(status_update)
        
        # Update trip status
        trip.status = new_status.value
        trip.updated_at = datetime.utcnow()
        
        self.db.commit()
        
        return f"Trip status updated to: {new_status.value}"

    def _report_issue(self, description: str) -> str:
        """Report an issue with the current trip"""
        trip = self.db.query(Trip).filter(
            Trip.driver_id == self.driver_id,
            Trip.status != TripStatus.COMPLETED.value
        ).order_by(Trip.created_at.desc()).first()
        
        if not trip:
            return "You don't have any active trips to report issues for."
        
        # Create issue
        issue = Issue(
            trip_id=trip.id,
            reported_by_id=self.driver_id,
            description=description,
            status="open"
        )
        self.db.add(issue)
        
        # Update trip status
        trip.status = TripStatus.ISSUE_REPORTED.value
        trip.updated_at = datetime.utcnow()
        
        # Create status update
        status_update = StatusUpdate(
            trip_id=trip.id,
            user_id=self.driver_id,
            status=TripStatus.ISSUE_REPORTED.value,
            notes=f"Issue reported: {description}"
        )
        self.db.add(status_update)
        
        self.db.commit()
        
        return f"Issue reported: {description}"

    def _update_location(self, latitude: float, longitude: float) -> str:
        """Update the current location of the driver"""
        trip = self.db.query(Trip).filter(
            Trip.driver_id == self.driver_id,
            Trip.status != TripStatus.COMPLETED.value
        ).order_by(Trip.created_at.desc()).first()
        
        if not trip:
            return "You don't have any active trips to update location for."
        
        # Create location
        location = Location(
            trip_id=trip.id,
            latitude=latitude,
            longitude=longitude
        )
        self.db.add(location)
        self.db.commit()
        
        return f"Location updated: {latitude}, {longitude}"

    def _get_trip_history(self) -> str:
        """Get the history of trips for the driver"""
        trips = self.db.query(Trip).filter(
            Trip.driver_id == self.driver_id
        ).order_by(Trip.created_at.desc()).limit(5).all()
        
        if not trips:
            return "You don't have any trip history."
        
        response = "Your recent trips:\n\n"
        for trip in trips:
            response += f"""
            Trip #{trip.id}:
            - Status: {trip.status}
            - Pickup: {trip.pickup_address}
            - Delivery: {trip.delivery_address}
            - Created: {trip.created_at.strftime('%Y-%m-%d')}
            """
        
        return response 