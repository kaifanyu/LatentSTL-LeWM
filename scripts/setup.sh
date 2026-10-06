#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
command -v uv >/dev/null || { echo 'Install uv before running setup.sh.' >&2; exit 1; }
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.11 .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 11), "Expected Python 3.11 in .venv"'
uv pip install --python .venv/bin/python -r requirements.lock.txt
mkdir -p third_party datasets .cache/libero
if [[ ! -d third_party/le-worldmodel/.git ]]; then
  git clone https://github.com/linzhongzi/le-worldmodel.git third_party/le-worldmodel
  git -C third_party/le-worldmodel checkout ca231f9f9d9ab041034b6d05e90b6e04bd6cff82
fi
if [[ ! -d third_party/LIBERO/.git ]]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
  git -C third_party/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
fi
# Packaging and PyTorch >=2.6 compatibility, matching existing local LIBERO.
touch third_party/LIBERO/libero/libero/__init__.py
.venv/bin/python - <<'PY'
from pathlib import Path
import yaml
root = Path.cwd()
p = root / 'third_party/LIBERO/libero/libero/benchmark/__init__.py'
p.write_text(p.read_text().replace('torch.load(init_states_path)', 'torch.load(init_states_path, weights_only=False)'))
base = root / 'third_party/LIBERO/libero/libero'
config = {'benchmark_root': str(base), 'datasets': str(root/'datasets')}
config.update({k: str(base/v) for k,v in {'assets':'assets','bddl_files':'bddl_files','init_states':'init_files'}.items()})
(root/'.cache/libero/config.yaml').write_text(yaml.safe_dump(config))
PY
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e third_party/LIBERO -e .
uv pip check --python .venv/bin/python
