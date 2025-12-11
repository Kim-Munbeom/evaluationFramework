"""
RAG (Retrieval-Augmented Generation) system evaluator.
Evaluates RAG systems using Faithfulness, Contextual Recall, and Answer Relevancy metrics.
"""
from typing import List, Dict, Any
from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from .base import BaseEvaluator, DeepEvalBaseLLM


class RAGEvaluator(BaseEvaluator):
    """Evaluator for RAG systems."""

    def __init__(
        self,
        model: DeepEvalBaseLLM,
        threshold: float = 0.7,
    ):
        """
        Initialize RAG evaluator.

        Args:
            model: DeepEval model instance
            threshold: Minimum score threshold for passing
        """
        super().__init__(model, threshold)

        # Initialize metrics
        self.faithfulness_metric = FaithfulnessMetric(
            threshold=threshold,
            model=model,
        )
        self.contextual_recall_metric = ContextualRecallMetric(
            threshold=threshold,
            model=model,
        )
        self.answer_relevancy_metric = AnswerRelevancyMetric(
            threshold=threshold,
            model=model,
        )

    def evaluate(self, test_cases: List[Any]) -> Dict[str, Any]:
        """
        Evaluate RAG test cases.

        Args:
            test_cases: List of RAGTestCase objects

        Returns:
            Dictionary containing evaluation results with scores and pass/fail status
        """
        results = {
            "total_cases": len(test_cases),
            "faithfulness_scores": [],
            "contextual_recall_scores": [],
            "answer_relevancy_scores": [],
            "individual_results": [],
        }

        for i, test_case in enumerate(test_cases):
            # Convert to DeepEval test case format
            llm_test_case = LLMTestCase(
                input=test_case.input,
                actual_output=test_case.actual_output,
                expected_output=test_case.expected_output,
                retrieval_context=test_case.context,
            )

            # Evaluate Faithfulness
            self.faithfulness_metric.measure(llm_test_case)
            faithfulness_score = self.faithfulness_metric.score

            # Evaluate Contextual Recall
            self.contextual_recall_metric.measure(llm_test_case)
            contextual_recall_score = self.contextual_recall_metric.score

            # Evaluate Answer Relevancy
            self.answer_relevancy_metric.measure(llm_test_case)
            answer_relevancy_score = self.answer_relevancy_metric.score

            # Store scores
            results["faithfulness_scores"].append(faithfulness_score)
            results["contextual_recall_scores"].append(contextual_recall_score)
            results["answer_relevancy_scores"].append(answer_relevancy_score)

            # Individual result
            individual_result = {
                "test_case_id": i,
                "input": test_case.input,
                "faithfulness": {
                    "score": faithfulness_score,
                    "passed": self.check_pass_threshold(faithfulness_score),
                    "reason": self.faithfulness_metric.reason,
                },
                "contextual_recall": {
                    "score": contextual_recall_score,
                    "passed": self.check_pass_threshold(contextual_recall_score),
                    "reason": self.contextual_recall_metric.reason,
                },
                "answer_relevancy": {
                    "score": answer_relevancy_score,
                    "passed": self.check_pass_threshold(answer_relevancy_score),
                    "reason": self.answer_relevancy_metric.reason,
                },
            }
            results["individual_results"].append(individual_result)

        # Calculate averages
        results["average_faithfulness"] = self.calculate_average_score(
            results["faithfulness_scores"]
        )
        results["average_contextual_recall"] = self.calculate_average_score(
            results["contextual_recall_scores"]
        )
        results["average_answer_relevancy"] = self.calculate_average_score(
            results["answer_relevancy_scores"]
        )

        # Overall average
        all_scores = (
            results["faithfulness_scores"]
            + results["contextual_recall_scores"]
            + results["answer_relevancy_scores"]
        )
        results["overall_average"] = self.calculate_average_score(all_scores)

        # Pass/fail determination
        results["passed"] = (
            results["overall_average"] >= self.threshold
            and results["average_faithfulness"] >= self.threshold
            and results["average_contextual_recall"] >= self.threshold
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
        report.append("RAG SYSTEM EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"\nTotal Test Cases: {results['total_cases']}")
        report.append(f"\nAverage Scores:")
        report.append(f"  - Faithfulness: {results['average_faithfulness']:.3f}")
        report.append(f"  - Contextual Recall: {results['average_contextual_recall']:.3f}")
        report.append(f"  - Answer Relevancy: {results['average_answer_relevancy']:.3f}")
        report.append(f"\nOverall Average: {results['overall_average']:.3f}")
        report.append(f"Status: {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
        report.append("\n" + "=" * 60)

        return "\n".join(report)
