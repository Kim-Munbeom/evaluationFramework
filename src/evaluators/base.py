"""
Base evaluator class for LLM system evaluation.
"""
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from deepeval.models import DeepEvalBaseLLM
from google import genai
from google.genai import types


class GeminiModel(DeepEvalBaseLLM):
    """Gemini model wrapper for DeepEval."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash-exp",
        api_key: Optional[str] = None,
    ):
        """
        Initialize Gemini model.

        Args:
            model: Gemini model name
            api_key: Google API key
        """
        self.model_name = model
        self.client = genai.Client(api_key=api_key)

    def load_model(self):
        """Load the model (not needed for API-based models)."""
        return self.client

    def generate(self, prompt: str) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: Input prompt

        Returns:
            Generated text response
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text

    async def a_generate(self, prompt: str) -> str:
        """
        Asynchronously generate a response from the model.

        Args:
            prompt: Input prompt

        Returns:
            Generated text response
        """
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_name


class BaseEvaluator(ABC):
    """Base class for all evaluators."""

    def __init__(
        self,
        model: DeepEvalBaseLLM,
        threshold: float = 0.7,
    ):
        """
        Initialize the base evaluator.

        Args:
            model: DeepEval model instance
            threshold: Minimum score threshold for passing
        """
        self.model = model
        self.threshold = threshold

    @abstractmethod
    def evaluate(self, test_cases: List[Any]) -> Dict[str, Any]:
        """
        Evaluate test cases.

        Args:
            test_cases: List of test case objects

        Returns:
            Dictionary containing evaluation results
        """
        pass

    def calculate_average_score(self, scores: List[float]) -> float:
        """
        Calculate average score from a list of scores.

        Args:
            scores: List of metric scores

        Returns:
            Average score
        """
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def check_pass_threshold(self, score: float) -> bool:
        """
        Check if a score passes the threshold.

        Args:
            score: Score to check

        Returns:
            True if score >= threshold
        """
        return score >= self.threshold
