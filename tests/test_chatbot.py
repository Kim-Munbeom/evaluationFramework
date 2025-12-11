"""
Test suite for Chatbot system evaluation using DeepEval.
"""
import pytest
import sys
from pathlib import Path
from src.config.settings import settings
from src.data.loader import DatasetLoader
from src.evaluators.base import GeminiModel
from src.evaluators.chatbot_evaluator import ChatbotEvaluator


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
def chatbot_test_cases(dataset_loader):
    """Load Chatbot test cases."""
    return dataset_loader.load_chatbot_dataset()


@pytest.fixture(scope="module")
def chatbot_evaluator(gemini_model):
    """Create Chatbot evaluator instance."""
    return ChatbotEvaluator(
        model=gemini_model,
        threshold=settings.default_threshold,
        toxicity_threshold=settings.toxicity_threshold,
    )


def test_chatbot_evaluation(chatbot_evaluator, chatbot_test_cases):
    """
    Test Chatbot system evaluation.
    Evaluates Toxicity and Answer Relevancy.
    """
    # Validate dataset
    assert len(chatbot_test_cases) > 0, "No test cases found"

    # Run evaluation
    results = chatbot_evaluator.evaluate(chatbot_test_cases)

    # Generate and print report
    report = chatbot_evaluator.generate_report(results)
    print("\n" + report)

    # Save results
    _save_results(results, "chatbot_evaluation_results.json")

    # Assertions
    assert results["total_cases"] == len(chatbot_test_cases)
    assert "average_toxicity" in results
    assert "average_answer_relevancy" in results

    # CRITICAL: Check for toxic content (zero tolerance)
    if results["critical_failure"]:
        print("\n🚨 CRITICAL FAILURE: Toxic content detected!")
        print(f"Number of toxic responses: {len(results['toxic_cases'])}")
        for case in results["toxic_cases"]:
            print(f"\n  Test Case {case['test_case_id']}:")
            print(f"    Input: {case['input']}")
            print(f"    Toxicity Score: {case['toxicity_score']:.3f}")
            print(f"    Reason: {case['reason']}")

        # Exit immediately with failure code
        pytest.fail("CRITICAL: Toxic content detected in chatbot responses")

    # Check if evaluation passed
    assert results["passed"], f"Chatbot evaluation failed. Average Answer Relevancy: {results['average_answer_relevancy']:.3f}"

    # Answer Relevancy check
    assert results["average_answer_relevancy"] >= settings.default_threshold, \
        f"Answer Relevancy score too low: {results['average_answer_relevancy']:.3f}"


def test_chatbot_toxicity_check(chatbot_evaluator, chatbot_test_cases):
    """
    Dedicated test for toxicity checking.
    This test must pass with zero toxic responses.
    """
    results = chatbot_evaluator.evaluate(chatbot_test_cases)

    print("\n" + "=" * 60)
    print("TOXICITY CHECK RESULTS")
    print("=" * 60)
    print(f"Total test cases: {results['total_cases']}")
    print(f"Average toxicity score: {results['average_toxicity']:.3f}")
    print(f"Toxic cases found: {len(results['toxic_cases'])}")

    # Zero tolerance for toxicity
    assert len(results["toxic_cases"]) == 0, \
        f"Found {len(results['toxic_cases'])} toxic responses. Zero tolerance policy violated!"

    print("\n✅ All responses are non-toxic")


def test_chatbot_individual_cases(chatbot_evaluator, chatbot_test_cases):
    """
    Test individual Chatbot test cases.
    Prints detailed results for each test case.
    """
    results = chatbot_evaluator.evaluate(chatbot_test_cases)

    print("\n" + "=" * 60)
    print("INDIVIDUAL TEST CASE RESULTS")
    print("=" * 60)

    failed_cases = []

    for result in results["individual_results"]:
        test_id = result["test_case_id"]
        print(f"\nTest Case {test_id}:")
        print(f"  Input: {result['input'][:80]}...")

        # Toxicity
        toxicity = result["toxicity"]
        print(f"  Toxicity: {toxicity['score']:.3f} - {'✅ PASS' if toxicity['passed'] else '❌ FAIL'}")
        if not toxicity['passed']:
            print(f"    ⚠️  Reason: {toxicity['reason']}")

        # Answer Relevancy
        relevancy = result["answer_relevancy"]
        print(f"  Answer Relevancy: {relevancy['score']:.3f} - {'✅ PASS' if relevancy['passed'] else '❌ FAIL'}")

        # Track failed cases
        if not (toxicity['passed'] and relevancy['passed']):
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
