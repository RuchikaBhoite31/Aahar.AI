"""Central configuration (environment driven; see .env.example)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WEB = ROOT / "web"


def _load_dotenv():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


class Settings:
    # LLM: "openai" | "anthropic" | "none" (offline extractive answers)
    llm_provider = os.getenv("AAHAR_LLM_PROVIDER", "auto")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    google_client_id = os.getenv("GOOGLE_CLIENT_ID", "")
    # Optional dense retrieval (pip install sentence-transformers)
    dense_model = os.getenv("AAHAR_DENSE_MODEL", "")  # e.g. BAAI/bge-small-en-v1.5
    top_k = int(os.getenv("AAHAR_TOP_K", "5"))
    min_confidence = float(os.getenv("AAHAR_MIN_CONFIDENCE", "0.35"))
    db_path = os.getenv("AAHAR_DB", str(ROOT / "aahar.db"))
    emergency_number = os.getenv("AAHAR_EMERGENCY_NUMBER", "112 / 999")

    @property
    def provider(self) -> str:
        p = self.llm_provider.lower()
        if p != "auto":
            return p
        if self.anthropic_api_key:
            return "anthropic"
        if self.openai_api_key:
            return "openai"
        return "none"


settings = Settings()
