#!/bin/bash
# Clone third-party dependencies into lib/ (shallow).
set -e
cd "$(dirname "$0")/../lib"
for r in QwenLM/Qwen3-ASR chimechallenge/chime-utils microsoft/NOTSOFAR1-Challenge; do
  [ -d "$(basename $r)" ] || git clone -q --depth 1 https://github.com/$r.git
done
