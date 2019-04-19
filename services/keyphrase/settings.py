import os
from dotenv import load_dotenv
load_dotenv()

ACTIVE_ENV = os.getenv("ACTIVE_ENV")
NATS_URL = os.getenv("NATS_URL")
DEFAULT_ENV = os.getenv("DEF_ENV")