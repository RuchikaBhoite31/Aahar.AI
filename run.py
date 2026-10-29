"""Run Aahar.AI:  python run.py  ->  http://localhost:8000"""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "aahar.api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
