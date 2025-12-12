"""
LLM 시스템 평가를 위한 기본 평가자 클래스
"""
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from deepeval.models import DeepEvalBaseLLM
from google import genai
from google.genai import types


class GeminiModel(DeepEvalBaseLLM):
    """DeepEval을 위한 Gemini 모델 래퍼"""

    def __init__(
        self,
        model: str = "gemini-2.0-flash-exp",
        api_key: Optional[str] = None,
    ):
        """
        Gemini 모델을 초기화합니다.

        Args:
            model: Gemini 모델 이름
            api_key: Google API 키
        """
        self.model_name = model
        self.client = genai.Client(api_key=api_key)

    def load_model(self):
        """모델을 로드합니다 (API 기반 모델에는 필요 없음)."""
        return self.client

    def generate(self, prompt: str) -> str:
        """
        모델로부터 응답을 생성합니다.

        Args:
            prompt: 입력 프롬프트

        Returns:
            생성된 텍스트 응답
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text

    async def a_generate(self, prompt: str) -> str:
        """
        모델로부터 비동기적으로 응답을 생성합니다.

        Args:
            prompt: 입력 프롬프트

        Returns:
            생성된 텍스트 응답
        """
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text

    def get_model_name(self) -> str:
        """모델 이름을 반환합니다."""
        return self.model_name


class BaseEvaluator(ABC):
    """모든 평가자의 기본 클래스"""

    def __init__(
        self,
        model: DeepEvalBaseLLM,
        threshold: float = 0.7,
    ):
        """
        기본 평가자를 초기화합니다.

        Args:
            model: DeepEval 모델 인스턴스
            threshold: 통과 최소 점수 임계값
        """
        self.model = model
        self.threshold = threshold

    @abstractmethod
    def evaluate(self, test_cases: List[Any]) -> Dict[str, Any]:
        """
        테스트 케이스를 평가합니다.

        Args:
            test_cases: 테스트 케이스 객체 리스트

        Returns:
            평가 결과를 담고 있는 딕셔너리
        """
        pass

    def calculate_average_score(self, scores: List[float]) -> float:
        """
        점수 리스트로부터 평균 점수를 계산합니다.

        Args:
            scores: 메트릭 점수 리스트

        Returns:
            평균 점수
        """
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def check_pass_threshold(self, score: float) -> bool:
        """
        점수가 임계값을 통과하는지 확인합니다.

        Args:
            score: 확인할 점수

        Returns:
            점수가 임계값 이상이면 True
        """
        return score >= self.threshold
