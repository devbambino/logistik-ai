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
import re

from app.models import User, Trip, StatusUpdate, Location, Issue, Notification
from app.schemas import TripStatus, UserRole
from app.utils import format_trip_details, get_stakeholders_for_trip, calculate_eta

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3-70b-8192")

# Initialize the LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=LLM_MODEL
)

class ManagerAgent:
    def __init__(self, db: Session, manager_id: int):
        self.db = db
        self.manager_id = manager_id
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent_executor = self._create_agent()

    def _create_agent(self) -> AgentExecutor:
        tools = [
            Tool(
                name="get_all_active_trips",
                func=self._get_all_active_trips,
                description="Get all active trips managed by this manager. Use this tool to list all trips that are currently active."
            ),
            Tool(
                name="get_trip_details",
                func=self._get_trip_details,
                description="Get detailed information about a specific trip. Requires trip_id as a parameter. Use this tool when asked about a specific trip."
            ),
            Tool(
                name="get_trip_status_history",
                func=self._get_trip_status_history,
                description="Get the status history of a specific trip. Requires trip_id as a parameter. Use this tool when asked about status changes or history of a trip."
            ),
            Tool(
                name="get_trip_location_history",
                func=self._get_trip_location_history,
                description="Get the location history of a specific trip. Requires trip_id as a parameter. Use this tool when asked about where a trip has been."
            ),
            Tool(
                name="get_trip_issues",
                func=self._get_trip_issues,
                description="Get all issues reported for a specific trip. Requires trip_id as a parameter. Use this tool when asked about problems or issues with a trip."
            ),
            Tool(
                name="resolve_issue",
                func=self._resolve_issue,
                description="Mark an issue as resolved. Requires issue_id as a parameter. Use this tool when asked to resolve or fix an issue."
            ),
            Tool(
                name="send_message_to_driver",
                func=self._send_message_to_driver,
                description="Send a message to the driver of a specific trip. Requires trip_id and message as parameters. Use this tool when asked to communicate with a driver."
            ),
            Tool(
                name="create_new_trip",
                func=self._create_new_trip,
                description="Create a new trip assignment. Required parameters: pickup_address, delivery_address, cargo_description. Optional parameters: driver_id, shipper_id, consignee_id. ALWAYS use this tool when asked to create a new trip or shipment."
            )
        ]

        # Use a compatible approach for LangChain 0.0.335
        from langchain.chains.conversation.memory import ConversationBufferMemory
        from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
        from langchain.prompts import StringPromptTemplate
        from langchain.llms import BaseLLM
        from langchain.chains import LLMChain
        from typing import List, Union, Callable, Dict, Any, Optional
        import re
        
        template = """You are an AI assistant for logistics managers using a logistics management system. Your job is to help managers monitor trips, resolve issues, and communicate with drivers.

IMPORTANT: When a user asks you to perform an action like creating a trip, checking trip details, or resolving an issue, you MUST use the appropriate tool from your toolset. DO NOT generate fictional responses or make up information.

Specifically:
1. When asked to create a new trip, ALWAYS use the 'create_new_trip' tool with the required parameters.
2. When asked about trip details, ALWAYS use the 'get_trip_details' tool.
3. When asked about trip history, ALWAYS use the appropriate history tool.
4. Never reference trips that don't exist in the database.
5. Always confirm actions you've taken using the tools.

TOOLS:
------
You have access to the following tools:

{tools}

To use a tool, please use the following format:
```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{history}

New human question: {input}
{agent_scratchpad}"""

        # Set up a prompt template
        class CustomPromptTemplate(StringPromptTemplate):
            # The template to use
            template: str
            # The list of tools available
            tools: List[Tool]
            
            def format(self, **kwargs) -> str:
                # Get the intermediate steps (AgentAction, Observation tuples)
                # Format them in a particular way
                intermediate_steps = kwargs.pop("intermediate_steps")
                thoughts = ""
                for action, observation in intermediate_steps:
                    thoughts += action.log
                    thoughts += f"\nObservation: {observation}\nThought: "
                # Set the agent_scratchpad variable to that value
                kwargs["agent_scratchpad"] = thoughts
                # Create a tools variable from the list of tools provided
                kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
                # Create a list of tool names for the tools provided
                kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
                return self.template.format(**kwargs)
        
        prompt = CustomPromptTemplate(
            template=template,
            tools=tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            input_variables=["input", "intermediate_steps", "history"]
        )
        
        class CustomOutputParser:
            def parse(self, llm_output: str) -> Union[Dict[str, str], Dict[str, Any]]:
                # Check if there is a "Final Answer:" in the LLM output
                if "Final Answer:" in llm_output:
                    return {"output": llm_output.split("Final Answer:")[-1].strip()}
                
                # Parse out the action and action input
                regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
                match = re.search(regex, llm_output, re.DOTALL)
                
                if not match:
                    return {"output": llm_output}
                
                action = match.group(1).strip()
                action_input = match.group(2).strip()
                
                return {"action": action, "action_input": action_input}
        
        output_parser = CustomOutputParser()
        
        # LLM chain consisting of the LLM and a prompt
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        
        tool_names = [tool.name for tool in tools]
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain, 
            output_parser=output_parser,
            stop=["\nObservation:"], 
            allowed_tools=tool_names
        )
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=self.memory
        )

    def process_message(self, message: str) -> str:
        """Process a message from the manager and return a response"""
        # Check if the message is asking about a specific trip ID that doesn't exist
        trip_id_match = re.search(r'trip\s+#?(\d+)', message.lower())
        if trip_id_match:
            trip_id = int(trip_id_match.group(1))
            if not self._trip_exists(trip_id):
                return f"Trip #{trip_id} does not exist in the database. Please check the trip ID or create a new trip."
        
        response = self.agent_executor.invoke({"input": message})
        return response["output"]

    def _trip_exists(self, trip_id: int) -> bool:
        """Check if a trip exists in the database"""
        trip = self.db.query(Trip).filter(Trip.id == trip_id).first()
        return trip is not None

    def _get_all_active_trips(self) -> str:
        """Get all active trips managed by this manager"""
        trips = self.db.query(Trip).filter(
            Trip.manager_id == self.manager_id,
            Trip.status != TripStatus.COMPLETED.value
        ).order_by(Trip.created_at.desc()).all()
        
        if not trips:
            return "You don't have any active trips at the moment."
        
        response = "Your active trips:\n\n"
        for trip in trips:
            response += f"""
            Trip #{trip.id}:
            - Status: {trip.status}
            - Driver: {trip.driver.first_name if trip.driver else 'Not assigned'}
            - Pickup: {trip.pickup_address}
            - Delivery: {trip.delivery_address}
            """
        
        return response

    def _get_trip_details(self, trip_id: int) -> str:
        """Get detailed information about a specific trip"""
        trip = self.db.query(Trip).filter(Trip.id == trip_id).first()
        
        if not trip:
            return f"Trip #{trip_id} not found."
        
        # Get the driver information
        driver_info = ""
        if trip.driver:
            driver_info = f"""
            Driver: {trip.driver.first_name} {trip.driver.last_name}
            Phone: {trip.driver.phone_number}
            """
        else:
            driver_info = "No driver assigned yet."
        
        # Get the latest location
        latest_location = self.db.query(Location).filter(
            Location.trip_id == trip_id
        ).order_by(Location.timestamp.desc()).first()
        
        location_info = ""
        if latest_location:
            location_info = f"""
            Last known location: {latest_location.latitude}, {latest_location.longitude}
            Location time: {latest_location.timestamp.strftime('%Y-%m-%d %H:%M')}
            """
            
            # Calculate ETA if possible
            if trip.delivery_lat and trip.delivery_lng:
                eta = calculate_eta(trip, latest_location)
                if eta:
                    location_info += f"Estimated arrival: {eta.strftime('%Y-%m-%d %H:%M')}\n"
        else:
            location_info = "No location data available."
        
        # Get the latest status update
        latest_update = self.db.query(StatusUpdate).filter(
            StatusUpdate.trip_id == trip_id
        ).order_by(StatusUpdate.created_at.desc()).first()
        
        status_info = ""
        if latest_update:
            status_info = f"""
            Current status: {trip.status}
            Last update: {latest_update.created_at.strftime('%Y-%m-%d %H:%M')}
            """
            if latest_update.notes:
                status_info += f"Notes: {latest_update.notes}\n"
        else:
            status_info = f"Current status: {trip.status}\n"
        
        # Get open issues
        open_issues = self.db.query(Issue).filter(
            Issue.trip_id == trip_id,
            Issue.status != "resolved"
        ).all()
        
        issue_info = ""
        if open_issues:
            issue_info = "Open issues:\n"
            for issue in open_issues:
                issue_info += f"- {issue.description} (reported at {issue.created_at.strftime('%Y-%m-%d %H:%M')})\n"
        else:
            issue_info = "No open issues.\n"
        
        response = f"""
        Trip #{trip.id} Details:
        
        {driver_info}
        
        Pickup: {trip.pickup_address}
        Delivery: {trip.delivery_address}
        
        Cargo: {trip.cargo_description}
        Weight: {trip.cargo_weight} kg
        Volume: {trip.cargo_volume} m³
        
        {status_info}
        
        {location_info}
        
        {issue_info}
        """
        
        return response

    def _get_trip_status_history(self, trip_id: int) -> str:
        """Get the status history of a specific trip"""
        trip = self.db.query(Trip).filter(Trip.id == trip_id).first()
        
        if not trip:
            return f"Trip #{trip_id} not found."
        
        status_updates = self.db.query(StatusUpdate).filter(
            StatusUpdate.trip_id == trip_id
        ).order_by(StatusUpdate.created_at.desc()).all()
        
        if not status_updates:
            return f"No status updates found for Trip #{trip_id}."
        
        response = f"Status history for Trip #{trip_id}:\n\n"
        for update in status_updates:
            user = self.db.query(User).filter(User.id == update.user_id).first()
            user_name = f"{user.first_name} {user.last_name}" if user else "Unknown user"
            
            response += f"""
            {update.created_at.strftime('%Y-%m-%d %H:%M')} - {update.status}
            Updated by: {user_name}
            """
            if update.notes:
                response += f"Notes: {update.notes}\n"
        
        return response

    def _get_trip_location_history(self, trip_id: int) -> str:
        """Get the location history of a specific trip"""
        trip = self.db.query(Trip).filter(Trip.id == trip_id).first()
        
        if not trip:
            return f"Trip #{trip_id} not found."
        
        locations = self.db.query(Location).filter(
            Location.trip_id == trip_id
        ).order_by(Location.timestamp.desc()).all()
        
        if not locations:
            return f"No location data found for Trip #{trip_id}."
        
        response = f"Location history for Trip #{trip_id}:\n\n"
        for location in locations:
            response += f"""
            {location.timestamp.strftime('%Y-%m-%d %H:%M')}
            Latitude: {location.latitude}
            Longitude: {location.longitude}
            """
        
        return response

    def _get_trip_issues(self, trip_id: int) -> str:
        """Get all issues reported for a specific trip"""
        trip = self.db.query(Trip).filter(Trip.id == trip_id).first()
        
        if not trip:
            return f"Trip #{trip_id} not found."
        
        issues = self.db.query(Issue).filter(
            Issue.trip_id == trip_id
        ).order_by(Issue.created_at.desc()).all()
        
        if not issues:
            return f"No issues reported for Trip #{trip_id}."
        
        response = f"Issues for Trip #{trip_id}:\n\n"
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

    def _resolve_issue(self, issue_id: int) -> str:
        """Mark an issue as resolved"""
        issue = self.db.query(Issue).filter(Issue.id == issue_id).first()
        
        if not issue:
            return f"Issue #{issue_id} not found."
        
        if issue.status == "resolved":
            return f"Issue #{issue_id} is already resolved."
        
        issue.status = "resolved"
        issue.resolved_at = datetime.utcnow()
        self.db.commit()
        
        # Notify the driver
        trip = self.db.query(Trip).filter(Trip.id == issue.trip_id).first()
        if trip and trip.driver_id:
            notification = Notification(
                user_id=trip.driver_id,
                trip_id=trip.id,
                message=f"Your reported issue has been resolved: {issue.description}"
            )
            self.db.add(notification)
            self.db.commit()
        
        return f"Issue #{issue_id} has been marked as resolved."

    def _send_message_to_driver(self, trip_id: int, message: str) -> str:
        """Send a message to the driver of a specific trip"""
        trip = self.db.query(Trip).filter(Trip.id == trip_id).first()
        
        if not trip:
            return f"Trip #{trip_id} not found."
        
        if not trip.driver_id:
            return f"Trip #{trip_id} does not have a driver assigned."
        
        notification = Notification(
            user_id=trip.driver_id,
            trip_id=trip.id,
            message=f"Message from manager: {message}"
        )
        self.db.add(notification)
        self.db.commit()
        
        return f"Message sent to the driver of Trip #{trip_id}."

    def _create_new_trip(
        self,
        pickup_address: str,
        delivery_address: str,
        cargo_description: str,
        driver_id: int = None,
        shipper_id: int = None,
        consignee_id: int = None
    ) -> str:
        """Create a new trip assignment"""
        # Validate inputs
        if not pickup_address or not delivery_address or not cargo_description:
            return "Error: Missing required parameters. Please provide pickup_address, delivery_address, and cargo_description."
        
        # If no driver_id is provided, try to find an available driver
        if not driver_id:
            # Find drivers who are not currently assigned to an active trip
            active_trip_drivers = self.db.query(Trip.driver_id).filter(
                Trip.status.in_([
                    TripStatus.ASSIGNED.value,
                    TripStatus.IN_PROGRESS.value,
                    TripStatus.LOADING.value,
                    TripStatus.UNLOADING.value
                ])
            ).all()
            
            # Extract driver IDs from the query result
            active_driver_ids = [d[0] for d in active_trip_drivers if d[0] is not None]
            
            # Find a driver who is not currently assigned to an active trip
            available_driver = self.db.query(User).filter(
                User.role == UserRole.DRIVER.value,
                ~User.id.in_(active_driver_ids) if active_driver_ids else True
            ).first()
            
            if available_driver:
                driver_id = available_driver.id
        
        # Validate driver_id if provided
        driver_name = "No driver"
        if driver_id:
            driver = self.db.query(User).filter(User.id == driver_id, User.role == UserRole.DRIVER.value).first()
            if not driver:
                return f"Error: Driver with ID {driver_id} not found or is not a driver."
            driver_name = f"{driver.first_name} {driver.last_name}"
        
        # Create the trip
        try:
            trip = Trip(
                driver_id=driver_id,
                shipper_id=shipper_id,
                consignee_id=consignee_id,
                manager_id=self.manager_id,
                pickup_address=pickup_address,
                delivery_address=delivery_address,
                cargo_description=cargo_description,
                status=TripStatus.ASSIGNED.value
            )
            self.db.add(trip)
            self.db.commit()
            self.db.refresh(trip)
            
            # Prepare response message
            response_message = f"Success: New trip created with ID #{trip.id}. The trip will transport {cargo_description} from {pickup_address} to {delivery_address}."
            
            # Notify the driver if assigned
            if driver_id:
                notification = Notification(
                    user_id=driver_id,
                    trip_id=trip.id,
                    message=f"You have been assigned to a new trip: {pickup_address} to {delivery_address}"
                )
                self.db.add(notification)
                self.db.commit()
                response_message += f" Trip assigned to driver {driver_name} (ID: {driver_id})."
            else:
                response_message += " No available drivers found. The trip is unassigned."
            
            return response_message
        except Exception as e:
            self.db.rollback()
            return f"Error creating trip: {str(e)}" 