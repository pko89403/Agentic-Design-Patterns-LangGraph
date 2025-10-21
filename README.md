

#LLM 서버 실행 명령어
no think api
```sh
llama-server --model Qwen_Qwen3-4B-Q4_K_M.gguf \
  --port 8080 \
  --threads 4 \
  --n-gpu-layers 12 \
  --ctx-size 2048 \
  --batch-size 4 \
  --reasoning-format none \
  --reasoning-budget 0 \
  --log-verbose
```
