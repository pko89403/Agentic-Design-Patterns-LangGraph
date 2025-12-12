# 에이전틱 디자인 패턴과 LangGraph

이 리포지토리는 LangChain과 LangGraph를 사용하여 에이전틱 디자인 패턴을 탐색하고 구현하는 방법을 배우는 한국인 파이썬 개발자를 위한 학습 공간입니다.

각 챕터는 Jupyter Notebook으로 구성되어 있으며, 특정 디자인 패턴이나 기술을 단계별로 학습할 수 있도록 안내합니다. 이 과정을 통해 스스로 추론하고, 학습하며, 복잡한 문제를 해결하는 자율 에이전트를 구축하는 데 필요한 핵심 원리를 익힐 수 있습니다.

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

LLM
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

EMBEDDING
```sh
llama-server -m ./embeddinggemma-300M-q8_0.gguf \
  --port 8081 \
  --embeddings
```

## 📚 튜토리얼 챕터

- **Chapter 0: [ChatGPT API 스타일로 llama.cpp 서버 호출하기](./chapter0.ipynb)**
  - `requests` 라이브러리를 사용하여 로컬 `llama.cpp` 서버와 상호작용하는 기본 방법을 학습합니다.
- **Chapter 1: [프롬프트 체이닝 (Prompt Chaining)](./chapter1-prompt-chaining.ipynb)**
  - 여러 프롬프트를 연결하여 더 복잡한 작업을 수행하는 방법을 학습합니다.
- **Chapter 2: [라우팅 (Routing)](./chapter2-routing.ipynb)**
  - 사용자 입력이나 이전 단계의 결과에 따라 동적으로 다음 단계를 결정하는 라우팅 에이전트를 학습합니다.
- **Chapter 3: [병렬화 (Parallelization)](./chapter3-parallelization.ipynb)**
  - 여러 작업을 동시에 실행하여 에이전트의 응답 속도를 높이는 방법을 학습합니다.
- **Chapter 4: [리플렉션 (Reflection)](./chapter4-reflection.ipynb)**
  - 에이전트가 자신의 출력을 검토하고 개선하는 리플렉션 패턴을 학습합니다.
- **Chapter 5: [도구 사용 (Tool Use)](./chapter5-tool-use.ipynb)**
  - 외부 도구를 호출하여 에이전트의 능력을 확장하는 방법을 학습합니다.
- **Chapter 6: [플래너 (Planner)](./chapter6-planner.ipynb)**
  - 복잡한 목표를 달성하기 위해 여러 단계를 계획하고 실행하는 플래너 패턴을 학습합니다.
- **Chapter 7: [다중 에이전트 협업 (Multi-Agent Collaboration)](./chapter7-multi-agent-collaboration.ipynb)**
  - 여러 전문 에이전트가 협력하여 단일 에이전트의 한계를 뛰어넘는 복잡한 문제를 해결하는 방법을 학습합니다.
- **Chapter 8: [메모리 관리 (Memory Management)](./chapter8-memory-management.ipynb)**
  - 에이전트가 대화의 맥락을 기억하고(단기 기억), 과거의 정보를 영구적으로 저장하고 검색(장기 기억)하는 방법을 학습합니다.
- **Chapter 9: [학습 및 적응 (Learning and Adaptation)](./chapter9-learning-and-adaptation.ipynb)**
  - 강화 학습(PPO, DPO)과 같은 기술을 통해 에이전트가 경험으로부터 학습하고 시간이 지남에 따라 스스로 성능을 개선하는 방법을 탐구합니다.
- **Chapter 10: [모델 컨텍스트 프로토콜 (Model Context Protocol)](./chapter10-model-context-protocol.ipynb)**
  - LLM이 외부 시스템, 데이터베이스, 도구와 표준화된 방식으로 상호작용할 수 있게 해주는 범용 어댑터인 MCP(Model Context Protocol)에 대해 학습합니다.
- **Chapter 11: [목표 설정 및 모니터링 (Goal Setting and Monitoring)](./chapter11-goal-setting-and-monitoring.ipynb)**
  - 에이전트가 명확한 목표를 설정하고, 그 목표를 달성하기 위한 진행 상황을 추적하며, 성공 여부를 스스로 평가하는 방법을 학습합니다.
- **Chapter 12: [예외 처리 및 복구 (Exception Handling and Recovery)](./chapter12-exception-handling-and-recovery.ipynb)**
  - 에이전트가 도구 오류, API 지연 등 예기치 않은 문제에 직면했을 때, 이를 감지하고 재시도, 대체 경로 사용 등의 전략으로 안정적으로 복구하는 방법을 학습합니다.
- **Chapter 13: [휴먼 인 더 루프 (Human-in-the-Loop)](./chapter13-human-in-the-loop.ipynb)**
  - 에이전트가 스스로 해결할 수 없는 문제에 직면했을 때, 사람의 개입을 요청하고 피드백을 받아 작업을 완료하는 방법을 학습합니다.
- **Chapter 14: [지식 검색 (Knowledge Retrieval - RAG)](./chapter14-knowledge-retrieval(rag).ipynb)**
  - 에이전트가 외부 데이터베이스나 문서에서 정보를 검색하여 최신 정보를 바탕으로 답변을 생성하는 RAG(Retrieval-Augmented Generation) 패턴을 학습합니다.
- **Chapter 15: [에이전트 간 통신 (Inter-Agent Communication - A2A)](./chapter15-inter-agent-communication(A2A).ipynb)**
  - 여러 에이전트가 서로 메시지를 주고받으며 협력하여 복잡한 문제를 해결하는 방법을 학습합니다.
- **Chapter 16: [리소스 인식 최적화 (Resource-Aware Optimization)](./chapter16-resource-aware-optiimization.ipynb)**
  - 에이전트가 작업의 비용과 이점을 평가하여 제한된 리소스(예: API 호출 수, 토큰 사용량) 내에서 최적의 결과를 도출하는 방법을 학습합니다.
- **Chapter 17: [추론 기술 (Reasoning Techniques)](./chapter17-reasoning-techniques.ipynb)**
  - CoT(Chain-of-Thought), ReAct(Reasoning-Action) 등과 같은 고급 추론 기술을 활용하여 에이전트가 더 복잡한 문제 해결 능력을 갖추는 방법을 탐구합니다.
- **Chapter 18: [가드레일 및 안전 패턴 (Guardrails and Safety Patterns)](./chapter18-guardrails_safety-patterns.ipynb)**
  - 에이전트가 유해하거나 원치 않는 행동을 하지 않도록 제한하는 안전 장치(가드레일)를 구현하는 방법을 학습합니다.
- **Chapter 19: [평가 및 모니터링 (Evaluation and Monitoring)](./chapter19-Evaluation-and-monitoring.ipynb)**
  - 에이전트의 성능을 정량적으로 평가하고, 운영 환경에서 지속적으로 모니터링하여 품질을 유지하는 방법을 학습합니다.
- **Chapter 20: [우선순위 지정 (Prioritization)](./chapter20-Prioritization.ipynb)**
  - 에이전트가 여러 작업이나 목표 중에서 중요도와 긴급성을 기준으로 우선순위를 동적으로 결정하는 방법을 학습합니다.
- **Chapter 21: [탐색 및 발견 (Exploration and Discovery)](./chapter21-Exploration-and-Discovery.ipynb)**
  - 에이전트가 새로운 도구, API, 정보 소스를 스스로 탐색하고 학습하여 자신의 능력을 확장해 나가는 방법을 학습합니다.

## 🎓 마무리

이것으로 `에이전틱 디자인 패턴과 LangGraph`의 모든 챕터가 마무리되었습니다. 이 튜토리얼을 통해 에이전트 설계의 핵심 원리를 이해하고, LangGraph를 사용하여 직접 구현하는 경험을 쌓으셨기를 바랍니다.

앞으로도 LLM과 에이전트 기술은 계속 발전할 것입니다. 여기에 소개된 패턴들을 기반으로 여러분만의 창의적인 아이디어를 더해 더욱 강력하고 지능적인 에이전트를 만들어보세요.


