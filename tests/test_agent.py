"""
Test suite for Agent system evaluation using DeepEval.
"""
import pytest
from pathlib import Path
from src.config.settings import settings
from src.data.loader import DatasetLoader
from src.evaluators.base import GeminiModel
from src.evaluators.agent_evaluator import AgentEvaluator


@pytest.fixture(scope="module")
def gemini_model():
    """Create Gemini model instance."""
    settings.validate_api_key()
    return GeminiModel(
        model=settings.gemini_model,
        api_key=settings.gemini_api_key,
    )


@pytest.fixture(scope="module")
def dataset_loader():
    """Create dataset loader instance."""
    return DatasetLoader(settings.datasets_dir)


@pytest.fixture(scope="module")
def agent_test_cases(dataset_loader):
    """Load Agent test cases."""
    return dataset_loader.load_agent_dataset()


@pytest.fixture(scope="module")
def agent_evaluator(gemini_model):
    """Create Agent evaluator instance."""
    return AgentEvaluator(
        model=gemini_model,
        threshold=settings.default_threshold,
    )


def test_agent_evaluation(agent_evaluator, agent_test_cases):
    """
    Test Agent system evaluation.
    Evaluates Correctness and Answer Relevancy.
    """
    # Validate dataset
    assert len(agent_test_cases) > 0, "No test cases found"

    # Run evaluation
    results = agent_evaluator.evaluate(agent_test_cases)

    # Generate and print report
    report = agent_evaluator.generate_report(results)
    print("\n" + report)

    # Save results
    _save_results(results, "agent_evaluation_results.json")

    # Assertions
    assert results["total_cases"] == len(agent_test_cases)
    assert "average_correctness" in results
    assert "average_answer_relevancy" in results
    assert "overall_average" in results

    # Check if evaluation passed
    assert results["passed"], f"Agent evaluation failed. Overall average: {results['overall_average']:.3f}"

    # Individual metric checks
    assert results["average_correctness"] >= settings.default_threshold, \
        f"Correctness score too low: {results['average_correctness']:.3f}"
    assert results["average_answer_relevancy"] >= settings.default_threshold, \
        f"Answer Relevancy score too low: {results['average_answer_relevancy']:.3f}"


def test_agent_individual_cases(agent_evaluator, agent_test_cases):
    """
    Test individual Agent test cases.
    Prints detailed results for each test case.
    """
    results = agent_evaluator.evaluate(agent_test_cases)

    print("\n" + "=" * 60)
    print("INDIVIDUAL TEST CASE RESULTS")
    print("=" * 60)

    failed_cases = []

    for result in results["individual_results"]:
        test_id = result["test_case_id"]
        print(f"\nTest Case {test_id}:")
        print(f"  Input: {result['input'][:80]}...")

        # Correctness
        correctness = result["correctness"]
        print(f"  Correctness: {correctness['score']:.3f} - {'✅ PASS' if correctness['passed'] else '❌ FAIL'}")

        # Answer Relevancy
        relevancy = result["answer_relevancy"]
        print(f"  Answer Relevancy: {relevancy['score']:.3f} - {'✅ PASS' if relevancy['passed'] else '❌ FAIL'}")

        # Track failed cases
        if not (correctness['passed'] and relevancy['passed']):
            failed_cases.append(test_id)

    if failed_cases:
        print(f"\n⚠️  Failed test cases: {failed_cases}")


def _save_results(results: dict, filename: str):
    """Save evaluation results to JSON file."""
    import json

    settings.ensure_directories()
    output_path = settings.report_dir / filename

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n📊 Results saved to: {output_path}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
