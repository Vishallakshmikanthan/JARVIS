import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    \"\"\"
    Global Configuration for IronMan AI (JARVIS + FRIDAY).
    Handles API keys, paths, and system settings.
    \"\"\"
    
    # --- Project Info ---
    APP_NAME = \"IronMan AI\"
    VERSION = \"1.0.0\"
    BASE_DIR = Path(__file__).resolve().parent
    
    # --- Persona Settings ---
    PRIMARY_PERSONA = os.getenv(\"PRIMARY_PERSONA\", \"JARVIS\")  # JARVIS or FRIDAY
    ASSISTANT_NAME = \"JARVIS\" if PRIMARY_PERSONA == \"JARVIS\" else \"FRIDAY\"
    USER_NAME = os.getenv(\"USER_NAME\", \"Sir\")
    
    # --- Ollama / LLM Settings ---
    OLLAMA_BASE_URL = os.getenv(\"OLLAMA_BASE_URL\", \"http://localhost:11434\")
    OLLAMA_MODEL = os.getenv(\"OLLAMA_MODEL\", \"llama3\")
    SYSTEM_PROMPT = f\"You are {ASSISTANT_NAME}, a highly intelligent AI assistant inspired by Iron Man's systems. You are helpful, efficient, and slightly witty.\"

    # --- Wake Word (Picovoice Porcupine) ---
    PORCUPINE_API_KEY = os.getenv(\"PORCUPINE_API_KEY\", \"\")
    WAKE_WORD_ACCESS_KEY = os.getenv(\"WAKE_WORD_ACCESS_KEY\", \"\")
    # Default wake word paths (adjust based on your .ppn files)
    WAKE_WORDS = [\"jarvis\", \"friday\"]
    SENSITIVITY = [0.5, 0.5] if len(WAKE_WORDS) > 1 else 0.5

    # --- API Keys ---
    SPOTIPY_CLIENT_ID = os.getenv(\"SPOTIPY_CLIENT_ID\", \"\")
    SPOTIPY_CLIENT_SECRET = os.getenv(\"SPOTIPY_CLIENT_SECRET\", \"\")
    SPOTIPY_REDIRECT_URI = os.getenv(\"SPOTIPY_REDIRECT_URI\", \"http://localhost:8888/callback\")
    
    NEWS_API_KEY = os.getenv(\"NEWS_API_KEY\", \"\")
    OPENWEATHER_API_KEY = os.getenv(\"OPENWEATHER_API_KEY\", \"\")
    
    # --- TTS / Voice Settings ---
    # Using Kokoro-ONNX path
    KOKORO_MODEL_PATH = os.getenv(\"KOKORO_MODEL_PATH\", str(BASE_DIR / \"models\" / \"kokoro-v0_19.onnx\"))
    KOKORO_VOICES_PATH = os.getenv(\"KOKORO_VOICES_PATH\", str(BASE_DIR / \"models\" / \"voices.bin\"))
    DEFAULT_VOICE = \"af_heart\" if PRIMARY_PERSONA == \"JARVIS\" else \"af_bella\"
    TTS_SPEED = 1.0

    # --- Location & Weather ---
    CITY = os.getenv(\"CITY\", \"Mumbai\")
    COUNTRY = os.getenv(\"COUNTRY\", \"India\")

    # --- Face Authentication ---
    FACE_AUTH_ENABLED = os.getenv(\"FACE_AUTH_ENABLED\", \"false\").lower() == \"true\"
    KNOWN_FACES_DIR = BASE_DIR / \"auth\" / \"known_faces\"
    FACE_RECOGNITION_THRESHOLD = 0.6

    # --- Database & Logs ---
    DB_PATH = os.getenv(\"DB_PATH\", str(BASE_DIR / \"core\" / \"database.db\"))
    LOG_DIR = BASE_DIR / \"logs\"
    LOG_LEVEL = os.getenv(\"LOG_LEVEL\", \"INFO\")

    # --- GUI Settings ---
    GUI_THEME = \"dark\"
    PRIMARY_COLOR = \"#00d4ff\" if PRIMARY_PERSONA == \"JARVIS\" else \"#ff0000\"

# Create instance
config = Config()

