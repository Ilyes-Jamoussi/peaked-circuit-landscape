#!/usr/bin/env bash
# Bootstrap a fresh GCP VM (or any Linux box) for the consolidation
# campaign. Clone both repositories side by side, then run this script:
#
#   git clone https://github.com/Ilyes-Jamoussi/peaked-circuit-landscape.git
#   git clone https://github.com/Ilyes-Jamoussi/peaked-circuits-pennylane.git
#   bash peaked-circuit-landscape/cloud/bootstrap.sh
#
# Then launch (inside tmux/nohup on the VM):
#   cd peaked-circuit-landscape
#   ../campaign-env/bin/python cloud/runner.py --mini            # gate
#   nohup ../campaign-env/bin/python cloud/runner.py --workers $(nproc) > campaign.out 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/../.."   # workspace root (parent of both repositories)

sudo apt-get update -y >/dev/null || true
sudo apt-get install -y python3-venv python3-dev >/dev/null || true

python3 -m venv campaign-env
./campaign-env/bin/pip install --quiet --upgrade pip
./campaign-env/bin/pip install --quiet -r peaked-circuit-landscape/requirements.txt

cd peaked-circuit-landscape
../campaign-env/bin/python test_pq.py
../campaign-env/bin/python cloud/runner.py --mini
# The `|| true` is not decoration: this line is informational, and a pipe that
# head closes early must not fail a gate that everything above it has passed.
{ ../campaign-env/bin/python cloud/runner.py --list || true; } | head -20
echo "bootstrap complete: mini dry-run passed; launch the full campaign in tmux/nohup."
