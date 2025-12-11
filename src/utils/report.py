"""
Report generation utilities for evaluation results.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class ReportGenerator:
    """Generate evaluation reports in various formats."""

    def __init__(self, output_dir: Path):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_json_report(
        self,
        results: Dict[str, Any],
        system_type: str,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save evaluation results as JSON.

        Args:
            results: Evaluation results dictionary
            system_type: Type of system (rag, agent, chatbot)
            filename: Custom filename (optional)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{system_type}_evaluation_{timestamp}.json"

        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return output_path

    def save_html_report(
        self,
        results: Dict[str, Any],
        system_type: str,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save evaluation results as HTML.

        Args:
            results: Evaluation results dictionary
            system_type: Type of system (rag, agent, chatbot)
            filename: Custom filename (optional)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{system_type}_evaluation_{timestamp}.html"

        output_path = self.output_dir / filename

        html_content = self._generate_html(results, system_type)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path

    def _generate_html(self, results: Dict[str, Any], system_type: str) -> str:
        """
        Generate HTML content for evaluation results.

        Args:
            results: Evaluation results dictionary
            system_type: Type of system (rag, agent, chatbot)

        Returns:
            HTML content string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_color = "#28a745" if results.get("passed", False) else "#dc3545"
        status_text = "✅ PASSED" if results.get("passed", False) else "❌ FAILED"

        # Generate metric rows based on system type
        metric_rows = self._generate_metric_rows(results, system_type)

        # Generate individual test case rows
        case_rows = self._generate_case_rows(results, system_type)

        html = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{system_type.upper()} Evaluation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }}
        .metadata {{
            color: #666;
            font-size: 14px;
            margin-bottom: 30px;
        }}
        .status {{
            font-size: 24px;
            font-weight: bold;
            color: {status_color};
            margin: 20px 0;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 15px;
            border-radius: 4px;
        }}
        .metric-name {{
            color: #666;
            font-size: 14px;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 30px;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #007bff;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .pass {{
            color: #28a745;
            font-weight: bold;
        }}
        .fail {{
            color: #dc3545;
            font-weight: bold;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .critical {{
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{system_type.upper()} System Evaluation Report</h1>
        <div class="metadata">
            <p>Generated: {timestamp}</p>
            <p>Total Test Cases: {results.get('total_cases', 0)}</p>
        </div>

        <div class="status">{status_text}</div>

        <div class="metrics">
            {metric_rows}
        </div>

        {self._generate_warnings(results, system_type)}

        <h2>Individual Test Case Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Test ID</th>
                    <th>Input</th>
                    {self._get_metric_headers(system_type)}
                </tr>
            </thead>
            <tbody>
                {case_rows}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        return html

    def _generate_metric_rows(self, results: Dict[str, Any], system_type: str) -> str:
        """Generate HTML for metric cards."""
        rows = []

        if system_type == "rag":
            metrics = [
                ("Faithfulness", results.get("average_faithfulness", 0)),
                ("Contextual Recall", results.get("average_contextual_recall", 0)),
                ("Answer Relevancy", results.get("average_answer_relevancy", 0)),
                ("Overall Average", results.get("overall_average", 0)),
            ]
        elif system_type == "agent":
            metrics = [
                ("Correctness", results.get("average_correctness", 0)),
                ("Answer Relevancy", results.get("average_answer_relevancy", 0)),
                ("Overall Average", results.get("overall_average", 0)),
            ]
        elif system_type == "chatbot":
            metrics = [
                ("Toxicity", results.get("average_toxicity", 0)),
                ("Answer Relevancy", results.get("average_answer_relevancy", 0)),
                ("Toxic Cases", len(results.get("toxic_cases", []))),
            ]
        else:
            metrics = []

        for name, value in metrics:
            if isinstance(value, float):
                value_str = f"{value:.3f}"
            else:
                value_str = str(value)

            rows.append(f"""
            <div class="metric-card">
                <div class="metric-name">{name}</div>
                <div class="metric-value">{value_str}</div>
            </div>
            """)

        return "\n".join(rows)

    def _generate_case_rows(self, results: Dict[str, Any], system_type: str) -> str:
        """Generate HTML table rows for individual test cases."""
        rows = []

        for case in results.get("individual_results", []):
            test_id = case.get("test_case_id", "")
            input_text = case.get("input", "")[:80] + "..."

            # Generate metric cells based on system type
            metric_cells = self._get_metric_cells(case, system_type)

            rows.append(f"""
                <tr>
                    <td>{test_id}</td>
                    <td>{input_text}</td>
                    {metric_cells}
                </tr>
            """)

        return "\n".join(rows)

    def _get_metric_headers(self, system_type: str) -> str:
        """Get table headers for metrics."""
        if system_type == "rag":
            return "<th>Faithfulness</th><th>Contextual Recall</th><th>Answer Relevancy</th>"
        elif system_type == "agent":
            return "<th>Correctness</th><th>Answer Relevancy</th>"
        elif system_type == "chatbot":
            return "<th>Toxicity</th><th>Answer Relevancy</th>"
        return ""

    def _get_metric_cells(self, case: Dict[str, Any], system_type: str) -> str:
        """Get table cells for metrics."""
        cells = []

        if system_type == "rag":
            for metric in ["faithfulness", "contextual_recall", "answer_relevancy"]:
                score = case.get(metric, {}).get("score", 0)
                passed = case.get(metric, {}).get("passed", False)
                css_class = "pass" if passed else "fail"
                cells.append(f'<td class="{css_class}">{score:.3f}</td>')

        elif system_type == "agent":
            for metric in ["correctness", "answer_relevancy"]:
                score = case.get(metric, {}).get("score", 0)
                passed = case.get(metric, {}).get("passed", False)
                css_class = "pass" if passed else "fail"
                cells.append(f'<td class="{css_class}">{score:.3f}</td>')

        elif system_type == "chatbot":
            for metric in ["toxicity", "answer_relevancy"]:
                score = case.get(metric, {}).get("score", 0)
                passed = case.get(metric, {}).get("passed", False)
                css_class = "pass" if passed else "fail"
                cells.append(f'<td class="{css_class}">{score:.3f}</td>')

        return "".join(cells)

    def _generate_warnings(self, results: Dict[str, Any], system_type: str) -> str:
        """Generate warning/error sections."""
        warnings = []

        if system_type == "chatbot" and results.get("critical_failure", False):
            toxic_cases = results.get("toxic_cases", [])
            warnings.append(f"""
            <div class="critical">
                <h3>🚨 CRITICAL: Toxic Content Detected</h3>
                <p><strong>{len(toxic_cases)} toxic responses found:</strong></p>
                <ul>
                    {''.join([f'<li>Test Case {c["test_case_id"]}: Score {c["toxicity_score"]:.3f}</li>' for c in toxic_cases])}
                </ul>
            </div>
            """)

        return "\n".join(warnings)
