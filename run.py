"""Run Aahar.AI:  python run.py  ->  http://localhost:8000"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("aahar.api:app", host="0.0.0.0", port=8000)
