"""
RAG (Retrieval-Augmented Generation) 시스템 평가자
Faithfulness, Contextual Recall, Answer Relevancy 메트릭을 사용하여 RAG 시스템을 평가합니다.
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
    """RAG 시스템 평가자"""

    def __init__(
        self,
        model: DeepEvalBaseLLM,
        threshold: float = 0.7,
    ):
        """
        RAG 평가자를 초기화합니다.

        Args:
            model: DeepEval 모델 인스턴스
            threshold: 통과 최소 점수 임계값
        """
        super().__init__(model, threshold)

        # 메트릭 초기화
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
        RAG 테스트 케이스를 평가합니다.

        Args:
            test_cases: RAGTestCase 객체 리스트

        Returns:
            점수와 통과/실패 상태를 포함하는 평가 결과 딕셔너리
        """
        results = {
            "total_cases": len(test_cases),
            "faithfulness_scores": [],
            "contextual_recall_scores": [],
            "answer_relevancy_scores": [],
            "individual_results": [],
        }

        for i, test_case in enumerate(test_cases):
            # DeepEval 테스트 케이스 형식으로 변환
            llm_test_case = LLMTestCase(
                input=test_case.input,
                actual_output=test_case.actual_output,
                expected_output=test_case.expected_output,
                retrieval_context=test_case.context,
            )

            # Faithfulness 평가
            self.faithfulness_metric.measure(llm_test_case)
            faithfulness_score = self.faithfulness_metric.score

            # Contextual Recall 평가
            self.contextual_recall_metric.measure(llm_test_case)
            contextual_recall_score = self.contextual_recall_metric.score

            # Answer Relevancy 평가
            self.answer_relevancy_metric.measure(llm_test_case)
            answer_relevancy_score = self.answer_relevancy_metric.score

            # 점수 저장
            results["faithfulness_scores"].append(faithfulness_score)
            results["contextual_recall_scores"].append(contextual_recall_score)
            results["answer_relevancy_scores"].append(answer_relevancy_score)

            # 개별 결과
            individual_result = {
                "test_case_id": i,
                "input": test_case.input,
                "actual_output": test_case.actual_output,
                "expected_output": test_case.expected_output,
                "context": test_case.context,
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

        # 평균 계산
        results["average_faithfulness"] = self.calculate_average_score(
            results["faithfulness_scores"]
        )
        results["average_contextual_recall"] = self.calculate_average_score(
            results["contextual_recall_scores"]
        )
        results["average_answer_relevancy"] = self.calculate_average_score(
            results["answer_relevancy_scores"]
        )

        # 전체 평균
        all_scores = (
            results["faithfulness_scores"]
            + results["contextual_recall_scores"]
            + results["answer_relevancy_scores"]
        )
        results["overall_average"] = self.calculate_average_score(all_scores)

        # 통과/실패 판정
        results["passed"] = (
            results["overall_average"] >= self.threshold
            and results["average_faithfulness"] >= self.threshold
            and results["average_contextual_recall"] >= self.threshold
            and results["average_answer_relevancy"] >= self.threshold
        )

        return results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        사람이 읽을 수 있는 평가 보고서를 생성합니다.

        Args:
            results: 평가 결과 딕셔너리

        Returns:
            포맷된 보고서 문자열
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
