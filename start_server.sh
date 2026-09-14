#!/usr/bin/env bash
set -euo pipefail
MODEL="${VPTRACK_MODEL:-jcwang0602/VPTracker}"
exec lmdeploy serve api_server "$MODEL"
