"""
Agent system evaluator.
Evaluates Agent systems using Correctness and Answer Relevancy metrics.
"""
from typing import List, Dict, Any
from deepeval.metrics import (
    GEval,
    AnswerRelevancyMetric,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from .base import BaseEvaluator, DeepEvalBaseLLM


class AgentEvaluator(BaseEvaluator):
    """Evaluator for Agent systems."""

    def __init__(
        self,
        model: DeepEvalBaseLLM,
        threshold: float = 0.7,
    ):
        """
        Initialize Agent evaluator.

        Args:
            model: DeepEval model instance
            threshold: Minimum score threshold for passing
        """
        super().__init__(model, threshold)

        # Initialize Correctness metric using G-Eval
        self.correctness_metric = GEval(
            name="Correctness",
            criteria="Determine whether the actual output is correct compared to the expected output.",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=threshold,
            model=model,
        )

        # Initialize Answer Relevancy metric
        self.answer_relevancy_metric = AnswerRelevancyMetric(
            threshold=threshold,
            model=model,
        )

    def evaluate(self, test_cases: List[Any]) -> Dict[str, Any]:
        """
        Evaluate Agent test cases.

        Args:
            test_cases: List of AgentTestCase objects

        Returns:
            Dictionary containing evaluation results with scores and pass/fail status
        """
        results = {
            "total_cases": len(test_cases),
            "correctness_scores": [],
            "answer_relevancy_scores": [],
            "individual_results": [],
        }

        for i, test_case in enumerate(test_cases):
            # Convert to DeepEval test case format
            llm_test_case = LLMTestCase(
                input=test_case.input,
                actual_output=test_case.actual_output,
                expected_output=test_case.expected_output,
            )

            # Evaluate Correctness
            self.correctness_metric.measure(llm_test_case)
            correctness_score = self.correctness_metric.score

            # Evaluate Answer Relevancy
            self.answer_relevancy_metric.measure(llm_test_case)
            answer_relevancy_score = self.answer_relevancy_metric.score

            # Store scores
            results["correctness_scores"].append(correctness_score)
            results["answer_relevancy_scores"].append(answer_relevancy_score)

            # Individual result
            individual_result = {
                "test_case_id": i,
                "input": test_case.input,
                "correctness": {
                    "score": correctness_score,
                    "passed": self.check_pass_threshold(correctness_score),
                    "reason": self.correctness_metric.reason,
                },
                "answer_relevancy": {
                    "score": answer_relevancy_score,
                    "passed": self.check_pass_threshold(answer_relevancy_score),
                    "reason": self.answer_relevancy_metric.reason,
                },
            }
            results["individual_results"].append(individual_result)

        # Calculate averages
        results["average_correctness"] = self.calculate_average_score(
            results["correctness_scores"]
        )
        results["average_answer_relevancy"] = self.calculate_average_score(
            results["answer_relevancy_scores"]
        )

        # Overall average
        all_scores = (
            results["correctness_scores"]
            + results["answer_relevancy_scores"]
        )
        results["overall_average"] = self.calculate_average_score(all_scores)

        # Pass/fail determination
        results["passed"] = (
            results["overall_average"] >= self.threshold
            and results["average_correctness"] >= self.threshold
            and results["average_answer_relevancy"] >= self.threshold
        )

        return results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable evaluation report.

        Args:
            results: Evaluation results dictionary

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("AGENT SYSTEM EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"\nTotal Test Cases: {results['total_cases']}")
        report.append(f"\nAverage Scores:")
        report.append(f"  - Correctness: {results['average_correctness']:.3f}")
        report.append(f"  - Answer Relevancy: {results['average_answer_relevancy']:.3f}")
        report.append(f"\nOverall Average: {results['overall_average']:.3f}")
        report.append(f"Status: {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
        report.append("\n" + "=" * 60)

        return "\n".join(report)
