import os
import sys
import requests
import json
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

def get_ngrok_url():
    """Get the public URL from ngrok."""
    try:
        # Connect to the ngrok API
        response = requests.get("http://localhost:4040/api/tunnels")
        data = response.json()
        
        # Extract the public URL
        for tunnel in data["tunnels"]:
            if tunnel["proto"] == "https":
                return tunnel["public_url"]
        
        print("No HTTPS tunnel found in ngrok.")
        return None
    except Exception as e:
        print(f"Error getting ngrok URL: {e}")
        return None

def update_env_file(ngrok_url):
    """Update the .env file with the ngrok URL."""
    try:
        # Read the current .env file
        with open(".env", "r") as f:
            lines = f.readlines()
        
        # Update or add the WEBHOOK_URL line
        webhook_line_found = False
        for i, line in enumerate(lines):
            if line.startswith("WEBHOOK_URL="):
                lines[i] = f"WEBHOOK_URL={ngrok_url}\n"
                webhook_line_found = True
                break
        
        if not webhook_line_found:
            lines.append(f"WEBHOOK_URL={ngrok_url}\n")
        
        # Write the updated .env file
        with open(".env", "w") as f:
            f.writelines(lines)
        
        print(f"Updated .env file with WEBHOOK_URL={ngrok_url}")
        return True
    except Exception as e:
        print(f"Error updating .env file: {e}")
        return False

def set_telegram_webhook(ngrok_url):
    """Set the Telegram webhook."""
    if not TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN not set in .env file.")
        return False
    
    webhook_url = f"{ngrok_url}/webhook/{TELEGRAM_BOT_TOKEN}"
    
    try:
        # Set the webhook
        response = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook",
            json={"url": webhook_url}
        )
        
        data = response.json()
        if data.get("ok"):
            print(f"Webhook set successfully: {webhook_url}")
            return True
        else:
            print(f"Error setting webhook: {data.get('description')}")
            return False
    except Exception as e:
        print(f"Error setting webhook: {e}")
        return False

def main():
    """Main function."""
    print("Setting up ngrok webhook...")
    
    # Get the ngrok URL
    ngrok_url = get_ngrok_url()
    if not ngrok_url:
        print("Failed to get ngrok URL. Make sure ngrok is running.")
        return
    
    # Update the .env file
    if not update_env_file(ngrok_url):
        print("Failed to update .env file.")
        return
    
    # Set the Telegram webhook
    if not set_telegram_webhook(ngrok_url):
        print("Failed to set Telegram webhook.")
        return
    
    print("Webhook setup complete!")
    print(f"Ngrok URL: {ngrok_url}")
    print(f"Webhook URL: {ngrok_url}/webhook/{TELEGRAM_BOT_TOKEN}")

if __name__ == "__main__":
    main() 