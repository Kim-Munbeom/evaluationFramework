"""
LLM 평가 프레임워크를 위한 설정
환경 변수를 로드하고 애플리케이션 전역 설정을 제공합니다.
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()


class Settings(BaseSettings):
    """환경 변수에서 로드된 애플리케이션 설정"""

    # Gemini API 설정
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

    # 평가 임계값
    default_threshold: float = float(os.getenv("DEFAULT_THRESHOLD", "0.7"))
    toxicity_threshold: float = float(os.getenv("TOXICITY_THRESHOLD", "0.0"))

    # 보고서 설정
    report_dir: Path = Path(os.getenv("REPORT_DIR", "./reports"))
    save_json: bool = os.getenv("SAVE_JSON", "true").lower() == "true"
    save_html: bool = os.getenv("SAVE_HTML", "true").lower() == "true"

    # 프로젝트 경로
    project_root: Path = Path(__file__).parent.parent.parent
    datasets_dir: Path = project_root / "datasets"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

    def validate_api_key(self) -> None:
        """Gemini API 키가 설정되었는지 검증합니다."""
        if not self.gemini_api_key or self.gemini_api_key == "your_api_key_here":
            raise ValueError(
                "GEMINI_API_KEY가 설정되지 않았습니다. "
                ".env 파일 또는 환경 변수에 설정해주세요."
            )

    def ensure_directories(self) -> None:
        """필요한 디렉토리가 존재하는지 확인합니다."""
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)


# 전역 설정 인스턴스
settings = Settings()
