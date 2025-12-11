"""
Configuration settings for the LLM evaluation framework.
Loads environment variables and provides application-wide settings.
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Gemini API Configuration
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

    # Evaluation Thresholds
    default_threshold: float = float(os.getenv("DEFAULT_THRESHOLD", "0.7"))
    toxicity_threshold: float = float(os.getenv("TOXICITY_THRESHOLD", "0.0"))

    # Report Settings
    report_dir: Path = Path(os.getenv("REPORT_DIR", "./reports"))
    save_json: bool = os.getenv("SAVE_JSON", "true").lower() == "true"
    save_html: bool = os.getenv("SAVE_HTML", "true").lower() == "true"

    # Project paths
    project_root: Path = Path(__file__).parent.parent.parent
    datasets_dir: Path = project_root / "datasets"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def validate_api_key(self) -> None:
        """Validate that the Gemini API key is set."""
        if not self.gemini_api_key or self.gemini_api_key == "your_api_key_here":
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Please set it in your .env file or environment variables."
            )

    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
