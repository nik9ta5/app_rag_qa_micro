exec llama-server \
  --model /models/gemma-3-1b-it.bf16.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --n-gpu-layers 99 \
  --ctx-size 8192 \
  --flash-attn on \
  --parallel 2 \
  --cont-batching \
  --chat-template gemma