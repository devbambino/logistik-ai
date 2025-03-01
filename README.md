# Logistik AI

An AI-powered logistics management system that uses Telegram for user interactions. This system helps manage the entire journey of truck drivers, from trip assignment to delivery completion, while keeping all stakeholders (managers, shippers, and consignees) informed.

## Features

- Telegram-based interaction for all users
- AI agent for natural language processing and decision making
- Real-time status updates and notifications
- Location tracking and sharing
- Issue reporting and resolution
- Trip management and documentation

## Tech Stack

- **Backend**: Python with FastAPI
- **AI Framework**: LangChain with Groq SDK (using llama3-70b-8192 model)
- **Database**: SQLite with SQLAlchemy (can be replaced with PostgreSQL in production)
- **Messaging**: Telegram Bot API
- **Development Tools**: Ngrok for webhook development

## System Architecture

### Core Components

1. **FastAPI Application**
   - RESTful API endpoints for all system operations
   - Webhook handler for Telegram bot integration
   - Background task processing

2. **Telegram Bot**
   - User interface for all stakeholders
   - Role-based command handling
   - Interactive buttons and location sharing

3. **AI Agents**
   - Role-specific AI assistants powered by LangChain and Groq
   - Natural language understanding and generation
   - Context-aware responses with conversation memory

4. **Database Models**
   - Users (drivers, managers, shippers, consignees)
   - Trips (assignments, statuses, locations)
   - Issues (reporting and resolution)
   - Notifications (messaging between stakeholders)

### User Roles

1. **Driver Agent**
   - Trip management and status updates
   - Location sharing
   - Issue reporting
   - Communication with other stakeholders

2. **Manager Agent**
   - Trip monitoring and assignment
   - Issue resolution
   - Communication with drivers, shippers, and consignees
   - Creating new trips

3. **Shipper Agent**
   - Shipment tracking
   - Communication with drivers and managers
   - Issue reporting and monitoring

4. **Consignee Agent**
   - Incoming shipment tracking
   - Estimated arrival time monitoring
   - Communication with drivers and shippers

### Data Flow

1. User sends a message via Telegram
2. Message is processed by the appropriate agent based on user role
3. Agent interacts with the database and generates a response
4. Response is sent back to the user via Telegram
5. Status updates and notifications are sent to relevant stakeholders

## Setup Instructions

### Prerequisites

- Python 3.9+
- Telegram account
- Groq API key

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/logistik-ai.git
   cd logistik-ai
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a Telegram bot:
   - Talk to [BotFather](https://t.me/botfather) on Telegram
   - Use the `/newbot` command to create a new bot
   - Copy the API token

5. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your Telegram Bot Token and Groq API Key

### Running the Application

1. Initialize the database (optional for test data):
   ```
   python scripts/init_db.py
   ```

2. Start the application:
   ```
   python run.py
   ```
   or
   ```
   uvicorn app.main:app --reload
   ```

3. Set up ngrok for webhook (in a separate terminal):
   ```
   ngrok http 8000
   ```

4. Configure the webhook:
   ```
   python scripts/setup_ngrok.py
   ```

## Project Structure

```
logistik-ai/
├── app/                      # Main application package
│   ├── agents/               # AI agents for different user roles
│   │   ├── __init__.py
│   │   ├── driver_agent.py
│   │   ├── manager_agent.py
│   │   ├── shipper_agent.py
│   │   └── consignee_agent.py
│   ├── __init__.py
│   ├── bot.py                # Telegram bot implementation
│   ├── database.py           # Database connection and session
│   ├── main.py               # FastAPI application and endpoints
│   ├── models.py             # SQLAlchemy database models
│   ├── schemas.py            # Pydantic schemas for validation
│   └── utils.py              # Utility functions
├── scripts/                  # Helper scripts
│   ├── init_db.py            # Database initialization
│   └── setup_ngrok.py        # Ngrok webhook configuration
├── .env.example              # Example environment variables
├── .gitignore                # Git ignore file
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
└── run.py                    # Application entry point
```

## Usage

1. Start a conversation with your bot on Telegram
2. Select your role (driver, manager, shipper, or consignee)
3. Follow the prompts to interact with the system
4. Use the provided buttons for quick responses

### Common Commands

- `/start` - Begin interaction and set your role
- `/help` - Get role-specific help information
- `/status` - Check current status (varies by role)
- `/setrole` - Change your current role

## Hackathon Project

This project was created for a hackathon and is intended for demonstration purposes. In a production environment, you would want to implement additional security measures, more robust error handling, and potentially replace SQLite with a more scalable database solution like PostgreSQL. 