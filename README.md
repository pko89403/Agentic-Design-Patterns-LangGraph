# 에이전틱 디자인 패턴과 LangGraph

이 리포지토리는 LangChain과 LangGraph를 사용하여 에이전틱 디자인 패턴을 탐색하고 구현하는 방법을 배우는 한국인 파이썬 개발자를 위한 학습 공간입니다.

각 챕터는 Jupyter Notebook으로 구성되어 있으며, 특정 디자인 패턴이나 기술을 단계별로 학습할 수 있도록 안내합니다.

## 🚀 시작하기

### 1. 사전 준비

이 프로젝트를 실행하려면 다음이 필요합니다:

- **Python 3.13 이상**
- **uv**: 파이썬 패키지 설치 및 관리를 위한 도구입니다. (`pip install uv`)

### 2. 의존성 설치

리포지토리를 클론한 후, 다음 명령어를 실행하여 필요한 라이브러리를 설치하세요.

```sh
uv sync
```

### 3. 로컬 LLM 서버 실행

이 프로젝트는 로컬 LLM을 사용합니다. `llama.cpp`를 사용하여 모델을 로드하고 API 서버를 실행해야 합니다.

다음은 `Qwen_Qwen2-7B-Instruct-Q4_K_M.gguf` 모델을 예시로 한 서버 실행 명령어입니다. 모델 파일 경로와 이름은 실제 환경에 맞게 수정해주세요.

```sh
llama-server --model Qwen_Qwen3-4B-Q4_K_M.gguf \
  --port 8080 \
  --threads 4 \
  --n-gpu-layers 12 \
  --chat-template-file qwen3-workaround.jinja \
  --ctx-size 2048 \
  --batch-size 4 \
  --reasoning-format none \
  --reasoning-budget 0 \
  --jinja \
  --log-verbose
```

## 📚 튜토리얼 챕터

- **Chapter 0: [ChatGPT API 스타일로 llama.cpp 서버 호출하기](./chapter0.ipynb)**
  - `requests` 라이브러리를 사용하여 로컬 `llama.cpp` 서버와 상호작용하는 기본 방법을 배웁니다.
- **Chapter 1: [프롬프트 체이닝 (Prompt Chaining)](./chapter1-prompt-chaining.ipynb)**
  - 여러 프롬프트를 연결하여 더 복잡한 작업을 수행하는 방법을 학습합니다.
- **Chapter 2: [라우팅 (Routing)](./chapter2-routing.ipynb)**
  - 사용자 입력이나 이전 단계의 결과에 따라 동적으로 다음 단계를 결정하는 라우팅 에이전트를 구축합니다.
- **Chapter 3: [병렬화 (Parallelization)](./chapter3-parallelization.ipynb)**
  - 여러 작업을 동시에 실행하여 에이전트의 응답 속도를 높이는 방법을 탐구합니다.
