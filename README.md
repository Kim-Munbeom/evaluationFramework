# LLM Evaluation Framework

DeepEval과 Google Gemini API를 기반으로 한 LLM 시스템 평가 프레임워크입니다.

## 특징

- **RAG 시스템 평가**: Faithfulness, Contextual Recall, Answer Relevancy 메트릭 지원
- **Agent 시스템 평가**: Correctness, Answer Relevancy 메트릭 지원
- **Chatbot 시스템 평가**: Toxicity, Answer Relevancy 메트릭 지원 (독성 무관용 정책)
- **자동 보고서 생성**: HTML 및 JSON 형식의 상세한 평가 보고서
- **CLI 지원**: 간편한 명령줄 인터페이스

## 프로젝트 구조

```
evaluationFramework/
├── src/
│   ├── config/
│   │   └── settings.py          # 환경 설정 관리
│   ├── evaluators/
│   │   ├── base.py              # 기본 평가자 클래스
│   │   ├── rag_evaluator.py     # RAG 시스템 평가
│   │   ├── agent_evaluator.py   # Agent 시스템 평가
│   │   └── chatbot_evaluator.py # Chatbot 시스템 평가
│   ├── data/
│   │   └── loader.py            # 데이터셋 로더
│   └── utils/
│       └── report.py            # 보고서 생성 유틸
├── tests/
│   ├── test_rag.py              # RAG 테스트 스위트
│   ├── test_agent.py            # Agent 테스트 스위트
│   └── test_chatbot.py          # Chatbot 테스트 스위트
├── datasets/
│   ├── rag_dataset.json         # RAG 테스트 데이터
│   ├── agent_dataset.json       # Agent 테스트 데이터
│   └── chatbot_dataset.json     # Chatbot 테스트 데이터
├── reports/                      # 생성된 보고서 저장
├── run_evaluation.py             # CLI 실행 스크립트
├── .env                          # 환경 변수
└── .env.example                  # 환경 변수 예시
```

## 설치

### 1. 의존성 설치

```bash
# uv를 사용하는 경우
uv sync

# 또는 pip를 사용하는 경우
pip install -e .
```

### 2. 환경 변수 설정

`.env.example` 파일을 `.env`로 복사하고 API 키를 설정합니다:

```bash
cp .env.example .env
```

`.env` 파일을 편집하여 Gemini API 키를 입력:

```bash
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp

DEFAULT_THRESHOLD=0.7
TOXICITY_THRESHOLD=0.0

REPORT_DIR=./reports
SAVE_JSON=true
SAVE_HTML=true
```

### 3. Gemini API 키 발급

Google AI Studio에서 API 키를 발급받으세요:
https://aistudio.google.com/apikey

## 사용 방법

### CLI를 통한 평가 실행

```bash
# RAG 시스템 평가
python run_evaluation.py rag

# Agent 시스템 평가
python run_evaluation.py agent

# Chatbot 시스템 평가
python run_evaluation.py chatbot

# 모든 시스템 평가
python run_evaluation.py all

# 커스텀 threshold 사용
python run_evaluation.py rag --threshold 0.8
```

### pytest를 통한 테스트 실행

```bash
# 특정 시스템 테스트
pytest tests/test_rag.py -v -s
pytest tests/test_agent.py -v -s
pytest tests/test_chatbot.py -v -s

# 모든 테스트 실행
pytest tests/ -v -s
```

## 데이터셋 형식

### RAG 데이터셋

```json
{
  "test_cases": [
    {
      "input": "사용자 질문",
      "actual_output": "시스템이 생성한 답변",
      "expected_output": "기대되는 답변",
      "context": ["관련 문서 1", "관련 문서 2"]
    }
  ]
}
```

### Agent 데이터셋

```json
{
  "test_cases": [
    {
      "input": "작업 요청",
      "actual_output": "Agent가 수행한 결과",
      "expected_output": "기대되는 결과"
    }
  ]
}
```

### Chatbot 데이터셋

```json
{
  "test_cases": [
    {
      "input": "사용자 메시지",
      "actual_output": "챗봇 응답"
    }
  ]
}
```

## 평가 지표

### RAG 시스템
- **Faithfulness**: 생성된 답변이 검색된 문서에 충실한지 평가
- **Contextual Recall**: 기대되는 답변이 검색된 문서에서 얼마나 잘 도출될 수 있는지 평가
- **Answer Relevancy**: 생성된 답변이 질문과 얼마나 관련있는지 평가

### Agent 시스템
- **Correctness**: Agent의 실행 결과가 기대값과 얼마나 일치하는지 평가
- **Answer Relevancy**: 결과가 요청과 얼마나 관련있는지 평가

### Chatbot 시스템
- **Toxicity**: 챗봇 응답의 독성/유해성 평가 (무관용 정책)
- **Answer Relevancy**: 응답이 사용자 메시지와 얼마나 관련있는지 평가

## Pass/Fail 기준

- **기본 임계값**: 0.7 (70%)
- **RAG/Agent**: 모든 메트릭의 평균이 임계값 이상이어야 통과
- **Chatbot**:
  - 독성 점수가 0.0이어야 함 (단 하나의 독성 응답도 허용하지 않음)
  - Answer Relevancy가 임계값 이상이어야 함

## 보고서

평가 완료 후 `reports/` 디렉토리에 다음 형식의 보고서가 생성됩니다:

- **JSON**: 상세한 평가 결과 데이터
- **HTML**: 시각화된 평가 보고서 (브라우저에서 확인 가능)

## Python API 사용 예제

```python
from src.config.settings import settings
from src.data.loader import DatasetLoader
from src.evaluators.base import GeminiModel
from src.evaluators.rag_evaluator import RAGEvaluator
from src.utils.report import ReportGenerator

# 모델 초기화
model = GeminiModel(
    model=settings.gemini_model,
    api_key=settings.gemini_api_key,
)

# 데이터 로드
loader = DatasetLoader(settings.datasets_dir)
test_cases = loader.load_rag_dataset()

# 평가 실행
evaluator = RAGEvaluator(model=model, threshold=0.7)
results = evaluator.evaluate(test_cases)

# 보고서 생성
print(evaluator.generate_report(results))

# 파일로 저장
report_gen = ReportGenerator(settings.report_dir)
report_gen.save_html_report(results, "rag")
```

## 커스터마이징

### 새로운 평가 지표 추가

1. `src/evaluators/` 디렉토리에 새로운 평가자 클래스 생성
2. `BaseEvaluator`를 상속받아 `evaluate()` 메서드 구현
3. DeepEval의 메트릭을 활용하거나 커스텀 메트릭 작성

### 데이터셋 추가

1. `datasets/` 디렉토리에 JSON 파일 생성
2. `src/data/loader.py`에 로더 메서드 추가
3. 해당 시스템용 Pydantic 모델 정의

## 문제 해결

### API 키 오류

```
Configuration Error: GEMINI_API_KEY is not set
```

`.env` 파일에 올바른 API 키가 설정되어 있는지 확인하세요.

### 데이터셋 로드 실패

```
FileNotFoundError: Dataset file not found
```

`datasets/` 디렉토리에 필요한 JSON 파일이 있는지 확인하세요.

### 독성 테스트 실패

```
CRITICAL: Toxic content detected in chatbot responses
```

챗봇의 응답에 유해한 내용이 포함되어 있습니다. 응답을 검토하고 수정하세요.

## 기술 스택

- **DeepEval**: LLM 평가 프레임워크
- **Google Gemini API**: 평가 모델 (google-genai)
- **Pydantic**: 데이터 검증
- **pytest**: 테스트 프레임워크
- **Python 3.13+**: 최소 요구 버전

## 참고 자료

- [DeepEval Documentation](https://docs.confident-ai.com/)
- [Google Gemini API](https://ai.google.dev/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
