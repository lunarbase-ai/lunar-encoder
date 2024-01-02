import uvicorn
from dotenv import load_dotenv
from pathlib import Path
import os

#TODO: Add CLI
if __name__ == "__main__":
    dotenv_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../config.env'))
    load_dotenv(dotenv_path=dotenv_path)
    uvicorn.run("api:app", host=os.getenv("API_ADDRESS"), port=int(os.getenv("API_PORT")), reload=False)
