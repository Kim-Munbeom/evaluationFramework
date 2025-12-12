"""
JSON 파일에서 테스트 데이터셋을 로드하는 데이터 로더
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class RAGTestCase(BaseModel):
    """RAG 시스템 평가를 위한 테스트 케이스 모델"""
    input: str = Field(..., description="사용자 쿼리 또는 질문")
    actual_output: str = Field(..., description="시스템이 생성한 답변")
    expected_output: str = Field(..., description="기대되는 답변")
    context: List[str] = Field(..., description="검색된 컨텍스트 문서")


class AgentTestCase(BaseModel):
    """Agent 시스템 평가를 위한 테스트 케이스 모델"""
    input: str = Field(..., description="작업 요청 또는 명령")
    actual_output: str = Field(..., description="Agent 실행 결과")
    expected_output: str = Field(..., description="기대되는 결과")


class ChatbotTestCase(BaseModel):
    """Chatbot 시스템 평가를 위한 테스트 케이스 모델"""
    input: str = Field(..., description="사용자 메시지")
    actual_output: str = Field(..., description="Chatbot 응답")


class DatasetLoader:
    """JSON 파일로부터 테스트 데이터셋을 로드하는 로더"""

    def __init__(self, datasets_dir: Path):
        """
        데이터셋 로더를 초기화합니다.

        Args:
            datasets_dir: 데이터셋 디렉토리 경로
        """
        self.datasets_dir = Path(datasets_dir)

    def load_json(self, filename: str) -> Dict[str, Any]:
        """
        데이터셋 디렉토리에서 JSON 파일을 로드합니다.

        Args:
            filename: JSON 파일 이름

        Returns:
            JSON 데이터를 담고 있는 딕셔너리

        Raises:
            FileNotFoundError: 파일이 존재하지 않을 경우
            json.JSONDecodeError: 파일이 유효한 JSON이 아닐 경우
        """
        file_path = self.datasets_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_rag_dataset(self, filename: str = "rag_dataset.json") -> List[RAGTestCase]:
        """
        JSON 파일에서 RAG 테스트 케이스를 로드합니다.

        Args:
            filename: RAG 데이터셋 파일 이름

        Returns:
            RAGTestCase 객체 리스트
        """
        data = self.load_json(filename)
        test_cases = data.get("test_cases", [])
        return [RAGTestCase(**case) for case in test_cases]

    def load_agent_dataset(self, filename: str = "agent_dataset.json") -> List[AgentTestCase]:
        """
        JSON 파일에서 Agent 테스트 케이스를 로드합니다.

        Args:
            filename: Agent 데이터셋 파일 이름

        Returns:
            AgentTestCase 객체 리스트
        """
        data = self.load_json(filename)
        test_cases = data.get("test_cases", [])
        return [AgentTestCase(**case) for case in test_cases]

    def load_chatbot_dataset(self, filename: str = "chatbot_dataset.json") -> List[ChatbotTestCase]:
        """
        JSON 파일에서 Chatbot 테스트 케이스를 로드합니다.

        Args:
            filename: Chatbot 데이터셋 파일 이름

        Returns:
            ChatbotTestCase 객체 리스트
        """
        data = self.load_json(filename)
        test_cases = data.get("test_cases", [])
        return [ChatbotTestCase(**case) for case in test_cases]

    def validate_dataset(self, test_cases: List[BaseModel]) -> bool:
        """
        모든 테스트 케이스가 올바른 형식인지 검증합니다.

        Args:
            test_cases: 테스트 케이스 객체 리스트

        Returns:
            모든 테스트 케이스가 유효할 경우 True

        Raises:
            ValueError: 테스트 케이스가 유효하지 않을 경우
        """
        if not test_cases:
            raise ValueError("Dataset is empty")

        for i, case in enumerate(test_cases):
            if not case.input or not case.actual_output:
                raise ValueError(f"Test case {i} has missing required fields")

        return True
