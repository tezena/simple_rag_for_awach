import os
from dotenv import load_dotenv


load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ['USER_AGENT'] = 'MyCustomUserAgent/1.0'
GOOGLE_API_KEY = os.getenv("API_KEY")
LANGCHAIN_API_KEY=os.getenv('LANGCHAIN_API_KEY')


