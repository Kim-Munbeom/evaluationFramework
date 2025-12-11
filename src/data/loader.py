"""
Data loader for loading test datasets from JSON files.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class RAGTestCase(BaseModel):
    """Test case model for RAG system evaluation."""
    input: str = Field(..., description="User query or question")
    actual_output: str = Field(..., description="System generated answer")
    expected_output: str = Field(..., description="Expected answer")
    context: List[str] = Field(..., description="Retrieved context documents")


class AgentTestCase(BaseModel):
    """Test case model for Agent system evaluation."""
    input: str = Field(..., description="Task request or command")
    actual_output: str = Field(..., description="Agent execution result")
    expected_output: str = Field(..., description="Expected result")


class ChatbotTestCase(BaseModel):
    """Test case model for Chatbot system evaluation."""
    input: str = Field(..., description="User message")
    actual_output: str = Field(..., description="Chatbot response")


class DatasetLoader:
    """Loader for test datasets from JSON files."""

    def __init__(self, datasets_dir: Path):
        """
        Initialize the dataset loader.

        Args:
            datasets_dir: Path to the datasets directory
        """
        self.datasets_dir = Path(datasets_dir)

    def load_json(self, filename: str) -> Dict[str, Any]:
        """
        Load a JSON file from the datasets directory.

        Args:
            filename: Name of the JSON file

        Returns:
            Dictionary containing the JSON data

        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file is not valid JSON
        """
        file_path = self.datasets_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_rag_dataset(self, filename: str = "rag_dataset.json") -> List[RAGTestCase]:
        """
        Load RAG test cases from a JSON file.

        Args:
            filename: Name of the RAG dataset file

        Returns:
            List of RAGTestCase objects
        """
        data = self.load_json(filename)
        test_cases = data.get("test_cases", [])
        return [RAGTestCase(**case) for case in test_cases]

    def load_agent_dataset(self, filename: str = "agent_dataset.json") -> List[AgentTestCase]:
        """
        Load Agent test cases from a JSON file.

        Args:
            filename: Name of the Agent dataset file

        Returns:
            List of AgentTestCase objects
        """
        data = self.load_json(filename)
        test_cases = data.get("test_cases", [])
        return [AgentTestCase(**case) for case in test_cases]

    def load_chatbot_dataset(self, filename: str = "chatbot_dataset.json") -> List[ChatbotTestCase]:
        """
        Load Chatbot test cases from a JSON file.

        Args:
            filename: Name of the Chatbot dataset file

        Returns:
            List of ChatbotTestCase objects
        """
        data = self.load_json(filename)
        test_cases = data.get("test_cases", [])
        return [ChatbotTestCase(**case) for case in test_cases]

    def validate_dataset(self, test_cases: List[BaseModel]) -> bool:
        """
        Validate that all test cases are properly formatted.

        Args:
            test_cases: List of test case objects

        Returns:
            True if all test cases are valid

        Raises:
            ValueError: If any test case is invalid
        """
        if not test_cases:
            raise ValueError("Dataset is empty")

        for i, case in enumerate(test_cases):
            if not case.input or not case.actual_output:
                raise ValueError(f"Test case {i} has missing required fields")

        return True
