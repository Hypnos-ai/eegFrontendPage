import os
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "collected_data"
    MODEL_DIR = BASE_DIR / "models"
    
    # Board settings
    BOARD_PORT = os.getenv("BOARD_PORT", "COM9")
    SAMPLE_RATE = 250
    NUM_CHANNELS = 6
    
    # Arduino settings
    ARDUINO_IP = os.getenv("ARDUINO_IP", "192.168.1.168")
    ARDUINO_PORT = int(os.getenv("ARDUINO_PORT", "80"))
    
    # Buffer settings
    BUFFER_SIZE = 1250 * 10
    WINDOW_SIZE = 1250
    OVERLAP_RATIO = 0.65
    
    @classmethod
    def ensure_directories(cls):
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True) 