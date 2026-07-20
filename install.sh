#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt
MAX_JOBS="${MAX_JOBS:-$(nproc)}" python -m pip install \
  flash-attn==2.8.3 \
  causal-conv1d==1.6.1 \
  flash-linear-attention==0.4.2 \
  --no-build-isolation
