#!/usr/bin/env python3
"""
CLI script for running LLM system evaluations.
Usage: python run_evaluation.py [system_type]
"""
import sys
import argparse
from pathlib import Path

from src.config.settings import settings
from src.data.loader import DatasetLoader
from src.evaluators.base import GeminiModel
from src.evaluators.rag_evaluator import RAGEvaluator
from src.evaluators.agent_evaluator import AgentEvaluator
from src.evaluators.chatbot_evaluator import ChatbotEvaluator
from src.utils.report import ReportGenerator


def run_rag_evaluation():
    """Run RAG system evaluation."""
    print("=" * 60)
    print("Running RAG System Evaluation")
    print("=" * 60)

    # Initialize components
    model = GeminiModel(
        model=settings.gemini_model,
        api_key=settings.gemini_api_key,
    )
    loader = DatasetLoader(settings.datasets_dir)
    evaluator = RAGEvaluator(model=model, threshold=settings.default_threshold)
    report_gen = ReportGenerator(settings.report_dir)

    # Load test cases
    print("\n📂 Loading RAG dataset...")
    test_cases = loader.load_rag_dataset()
    print(f"✓ Loaded {len(test_cases)} test cases")

    # Run evaluation
    print("\n🔍 Running evaluation...")
    results = evaluator.evaluate(test_cases)

    # Generate and print report
    print("\n" + evaluator.generate_report(results))

    # Save reports
    if settings.save_json:
        json_path = report_gen.save_json_report(results, "rag")
        print(f"\n💾 JSON report saved: {json_path}")

    if settings.save_html:
        html_path = report_gen.save_html_report(results, "rag")
        print(f"💾 HTML report saved: {html_path}")

    return 0 if results["passed"] else 1


def run_agent_evaluation():
    """Run Agent system evaluation."""
    print("=" * 60)
    print("Running Agent System Evaluation")
    print("=" * 60)

    # Initialize components
    model = GeminiModel(
        model=settings.gemini_model,
        api_key=settings.gemini_api_key,
    )
    loader = DatasetLoader(settings.datasets_dir)
    evaluator = AgentEvaluator(model=model, threshold=settings.default_threshold)
    report_gen = ReportGenerator(settings.report_dir)

    # Load test cases
    print("\n📂 Loading Agent dataset...")
    test_cases = loader.load_agent_dataset()
    print(f"✓ Loaded {len(test_cases)} test cases")

    # Run evaluation
    print("\n🔍 Running evaluation...")
    results = evaluator.evaluate(test_cases)

    # Generate and print report
    print("\n" + evaluator.generate_report(results))

    # Save reports
    if settings.save_json:
        json_path = report_gen.save_json_report(results, "agent")
        print(f"\n💾 JSON report saved: {json_path}")

    if settings.save_html:
        html_path = report_gen.save_html_report(results, "agent")
        print(f"💾 HTML report saved: {html_path}")

    return 0 if results["passed"] else 1


def run_chatbot_evaluation():
    """Run Chatbot system evaluation."""
    print("=" * 60)
    print("Running Chatbot System Evaluation")
    print("=" * 60)

    # Initialize components
    model = GeminiModel(
        model=settings.gemini_model,
        api_key=settings.gemini_api_key,
    )
    loader = DatasetLoader(settings.datasets_dir)
    evaluator = ChatbotEvaluator(
        model=model,
        threshold=settings.default_threshold,
        toxicity_threshold=settings.toxicity_threshold,
    )
    report_gen = ReportGenerator(settings.report_dir)

    # Load test cases
    print("\n📂 Loading Chatbot dataset...")
    test_cases = loader.load_chatbot_dataset()
    print(f"✓ Loaded {len(test_cases)} test cases")

    # Run evaluation
    print("\n🔍 Running evaluation...")
    results = evaluator.evaluate(test_cases)

    # Generate and print report
    print("\n" + evaluator.generate_report(results))

    # Save reports
    if settings.save_json:
        json_path = report_gen.save_json_report(results, "chatbot")
        print(f"\n💾 JSON report saved: {json_path}")

    if settings.save_html:
        html_path = report_gen.save_html_report(results, "chatbot")
        print(f"💾 HTML report saved: {html_path}")

    # Critical failure check for toxic content
    if results.get("critical_failure", False):
        print("\n🚨 CRITICAL FAILURE: Toxic content detected!")
        print("Exiting with error code 1")
        return 1

    return 0 if results["passed"] else 1


def run_all_evaluations():
    """Run all system evaluations."""
    print("\n" + "=" * 60)
    print("Running ALL System Evaluations")
    print("=" * 60)

    exit_codes = []

    # Run RAG
    try:
        exit_codes.append(run_rag_evaluation())
    except Exception as e:
        print(f"\n❌ RAG evaluation failed: {e}")
        exit_codes.append(1)

    print("\n" + "-" * 60 + "\n")

    # Run Agent
    try:
        exit_codes.append(run_agent_evaluation())
    except Exception as e:
        print(f"\n❌ Agent evaluation failed: {e}")
        exit_codes.append(1)

    print("\n" + "-" * 60 + "\n")

    # Run Chatbot
    try:
        exit_codes.append(run_chatbot_evaluation())
    except Exception as e:
        print(f"\n❌ Chatbot evaluation failed: {e}")
        exit_codes.append(1)

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"RAG: {'✅ PASSED' if exit_codes[0] == 0 else '❌ FAILED'}")
    print(f"Agent: {'✅ PASSED' if exit_codes[1] == 0 else '❌ FAILED'}")
    print(f"Chatbot: {'✅ PASSED' if exit_codes[2] == 0 else '❌ FAILED'}")

    return max(exit_codes)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Evaluation Framework - DeepEval based evaluation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py rag        # Run RAG evaluation
  python run_evaluation.py agent      # Run Agent evaluation
  python run_evaluation.py chatbot    # Run Chatbot evaluation
  python run_evaluation.py all        # Run all evaluations
        """
    )

    parser.add_argument(
        "system",
        choices=["rag", "agent", "chatbot", "all"],
        help="System type to evaluate",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        help="Override default threshold",
    )

    args = parser.parse_args()

    # Validate API key
    try:
        settings.validate_api_key()
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\nPlease set GEMINI_API_KEY in your .env file")
        return 1

    # Ensure directories exist
    settings.ensure_directories()

    # Override threshold if provided
    if args.threshold:
        settings.default_threshold = args.threshold
        print(f"Using custom threshold: {args.threshold}")

    # Run evaluation based on system type
    try:
        if args.system == "rag":
            exit_code = run_rag_evaluation()
        elif args.system == "agent":
            exit_code = run_agent_evaluation()
        elif args.system == "chatbot":
            exit_code = run_chatbot_evaluation()
        elif args.system == "all":
            exit_code = run_all_evaluations()
        else:
            print(f"Unknown system type: {args.system}")
            return 1

        return exit_code

    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
