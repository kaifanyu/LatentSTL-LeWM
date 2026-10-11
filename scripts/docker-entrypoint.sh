#!/usr/bin/env bash
set -euo pipefail
cd /workspace
source scripts/env.sh
mkdir -p "$LIBERO_CONFIG_PATH"

# The host cache mount hides the config generated during the image build.
python - <<'PY'
from pathlib import Path
import os
import yaml

root = Path('/workspace')
base = root / 'third_party/LIBERO/libero/libero'
config = {'benchmark_root': str(base), 'datasets': str(root / 'datasets')}
config.update({key: str(base / relative) for key, relative in {
    'assets': 'assets', 'bddl_files': 'bddl_files', 'init_states': 'init_files'
}.items()})
path = Path(os.environ['LIBERO_CONFIG_PATH']) / 'config.yaml'
path.write_text(yaml.safe_dump(config))
PY

exec "$@"
