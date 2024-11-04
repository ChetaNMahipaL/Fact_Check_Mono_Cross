import dotenv
import os

# Load environment variables from the .env file
dotenv.load_dotenv()

# Now you can access your API key like this
openai_api_key = os.getenv("OPENAI_API_KEY")
