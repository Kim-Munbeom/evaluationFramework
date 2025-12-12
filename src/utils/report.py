"""
평가 결과를 위한 보고서 생성 유틸리티
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class ReportGenerator:
    """다양한 형식의 평가 보고서를 생성합니다."""

    def __init__(self, output_dir: Path):
        """
        보고서 생성기를 초기화합니다.

        Args:
            output_dir: 보고서를 저장할 디렉토리
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
        평가 결과를 JSON으로 저장합니다.

        Args:
            results: 평가 결과 딕셔너리
            system_type: 시스템 타입 (rag, agent, chatbot)
            filename: 커스텀 파일명 (선택사항)

        Returns:
            저장된 파일 경로
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
        평가 결과를 HTML로 저장합니다.

        Args:
            results: 평가 결과 딕셔너리
            system_type: 시스템 타입 (rag, agent, chatbot)
            filename: 커스텀 파일명 (선택사항)

        Returns:
            저장된 파일 경로
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
        평가 결과를 위한 HTML 콘텐츠를 생성합니다.

        Args:
            results: 평가 결과 딕셔너리
            system_type: 시스템 타입 (rag, agent, chatbot)

        Returns:
            HTML 콘텐츠 문자열
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_color = "#28a745" if results.get("passed", False) else "#dc3545"
        status_text = "✅ 통과" if results.get("passed", False) else "❌ 실패"

        # 시스템 타입에 따라 메트릭 행 생성
        metric_rows = self._generate_metric_rows(results, system_type)

        # 개별 테스트 케이스 행 생성
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
        .case-row:hover {{
            background-color: #e3f2fd !important;
        }}
        .detail-row td {{
            border: none;
        }}
    </style>
    <script>
        function toggleDetail(detailId) {{
            const detailRow = document.getElementById(detailId);
            if (detailRow.style.display === 'none') {{
                detailRow.style.display = 'table-row';
            }} else {{
                detailRow.style.display = 'none';
            }}
        }}
    </script>
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
        """메트릭 카드를 위한 HTML을 생성합니다."""
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
        """개별 테스트 케이스를 위한 HTML 테이블 행을 생성합니다."""
        rows = []

        for case in results.get("individual_results", []):
            test_id = case.get("test_case_id", "")
            input_text = case.get("input", "")[:80] + "..." if len(case.get("input", "")) > 80 else case.get("input", "")

            # 시스템 타입에 따라 메트릭 셀 생성
            metric_cells = self._get_metric_cells(case, system_type)

            # 전체 정보를 포함하는 상세 행 생성
            detail_row = self._generate_detail_row(case, system_type, test_id)

            rows.append(f"""
                <tr class="case-row" onclick="toggleDetail('detail-{test_id}')" style="cursor: pointer;">
                    <td>{test_id}</td>
                    <td>{input_text} <span style="color: #007bff;">▼</span></td>
                    {metric_cells}
                </tr>
                {detail_row}
            """)

        return "\n".join(rows)

    def _generate_detail_row(self, case: Dict[str, Any], system_type: str, test_id: str) -> str:
        """테스트 케이스를 위한 확장 가능한 상세 행을 생성합니다."""
        input_full = case.get("input", "").replace("\n", "<br>")
        actual_output = case.get("actual_output", "").replace("\n", "<br>")
        expected_output = case.get("expected_output", "")
        context = case.get("context", [])

        # 시스템 타입에 따라 상세 콘텐츠 구성
        detail_content = f"""
            <div style="margin-bottom: 15px;">
                <strong>입력:</strong><br>
                <div style="background: #f8f9fa; padding: 10px; border-radius: 4px; margin-top: 5px;">
                    {input_full}
                </div>
            </div>
            <div style="margin-bottom: 15px;">
                <strong>실제 출력:</strong><br>
                <div style="background: #f8f9fa; padding: 10px; border-radius: 4px; margin-top: 5px;">
                    {actual_output}
                </div>
            </div>
        """

        if expected_output:
            expected_output = expected_output.replace("\n", "<br>")
            detail_content += f"""
            <div style="margin-bottom: 15px;">
                <strong>예상 출력:</strong><br>
                <div style="background: #fff3cd; padding: 10px; border-radius: 4px; margin-top: 5px;">
                    {expected_output}
                </div>
            </div>
            """

        if context and system_type == "rag":
            context_html = "<br><br>".join([f"<li>{ctx.replace('<', '&lt;').replace('>', '&gt;')}</li>" for ctx in context])
            detail_content += f"""
            <div style="margin-bottom: 15px;">
                <strong>컨텍스트 (검색된 문서):</strong><br>
                <ul style="background: #e7f3ff; padding: 15px 15px 15px 35px; border-radius: 4px; margin-top: 5px;">
                    {context_html}
                </ul>
            </div>
            """

        # 메트릭 이유 추가
        detail_content += self._generate_metric_reasons(case, system_type)

        return f"""
        <tr id="detail-{test_id}" class="detail-row" style="display: none;">
            <td colspan="10" style="background: #f0f0f0; padding: 20px;">
                {detail_content}
            </td>
        </tr>
        """

    def _generate_metric_reasons(self, case: Dict[str, Any], system_type: str) -> str:
        """메트릭 평가 이유를 위한 HTML을 생성합니다."""
        reasons_html = """
            <div style="margin-top: 20px;">
                <strong>평가 이유:</strong><br>
        """

        metrics = []
        if system_type == "rag":
            metrics = ["faithfulness", "contextual_recall", "answer_relevancy"]
        elif system_type == "agent":
            metrics = ["correctness", "answer_relevancy"]
        elif system_type == "chatbot":
            metrics = ["toxicity", "answer_relevancy"]

        for metric in metrics:
            metric_data = case.get(metric, {})
            reason = metric_data.get("reason", "")
            score = metric_data.get("score", 0)
            passed = metric_data.get("passed", False)

            if reason:
                metric_name = metric.replace("_", " ").title()
                status_color = "#28a745" if passed else "#dc3545"
                status_icon = "✅" if passed else "❌"

                reasons_html += f"""
                <div style="background: #ffffff; border-left: 4px solid {status_color}; padding: 12px; margin: 10px 0; border-radius: 4px;">
                    <div style="font-weight: bold; color: {status_color}; margin-bottom: 5px;">
                        {status_icon} {metric_name} (Score: {score:.3f})
                    </div>
                    <div style="color: #555; font-size: 14px;">
                        {reason.replace('<', '&lt;').replace('>', '&gt;')}
                    </div>
                </div>
                """

        reasons_html += "</div>"
        return reasons_html

    def _get_metric_headers(self, system_type: str) -> str:
        """메트릭을 위한 테이블 헤더를 가져옵니다."""
        if system_type == "rag":
            return "<th>Faithfulness</th><th>Contextual Recall</th><th>Answer Relevancy</th>"
        elif system_type == "agent":
            return "<th>Correctness</th><th>Answer Relevancy</th>"
        elif system_type == "chatbot":
            return "<th>Toxicity</th><th>Answer Relevancy</th>"
        return ""

    def _get_metric_cells(self, case: Dict[str, Any], system_type: str) -> str:
        """메트릭을 위한 테이블 셀을 가져옵니다."""
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
        """경고/오류 섹션을 생성합니다."""
        warnings = []

        if system_type == "chatbot" and results.get("critical_failure", False):
            toxic_cases = results.get("toxic_cases", [])
            warnings.append(f"""
            <div class="critical">
                <h3>🚨 치명적: Toxic 콘텐츠 발견</h3>
                <p><strong>{len(toxic_cases)}개의 toxic 응답이 발견되었습니다:</strong></p>
                <ul>
                    {''.join([f'<li>테스트 케이스 {c["test_case_id"]}: 점수 {c["toxicity_score"]:.3f}</li>' for c in toxic_cases])}
                </ul>
            </div>
            """)

        return "\n".join(warnings)
