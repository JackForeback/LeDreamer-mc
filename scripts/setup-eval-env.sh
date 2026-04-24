#!/bin/bash
# Setup script for the eval-env venv (Python 3.9) with all dreamer4 + minerl deps.
# Assumes eval-env/ already exists with Python 3.9 and core packages installed
# (torch, minerl, numpy, etc.). This script (a) uninstalls the typing backport,
# (b) installs/pins the packages that must bypass the resolver, and (c) patches
# two site-packages files whose Python 3.10+ syntax breaks on Python 3.9.
#
# IMPORTANT: Installs come BEFORE source patches. pip reinstalls a package
# verbatim even when the version already matches, which silently wipes any
# prior in-place patch. Patching last guarantees the patches survive.
#
# Run from the repo root:
#   bash scripts/setup-eval-venv.sh
#
#   FIXME Currently ignores minerl and custom deps needed for dreamer4 (train env cmd).
set -e

VENV_DIR="./eval-env"
PIP="${VENV_DIR}/bin/pip"
PYTHON="${VENV_DIR}/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: $VENV_DIR does not look like a venv (no $PYTHON)." >&2
    echo "Create it first with: python3.9 -m venv $VENV_DIR" >&2
    exit 1
fi

echo "=== Step 1: Remove typing backport (shadows stdlib typing on 3.9) ==="
$PIP uninstall typing -y 2>/dev/null || echo "  Not installed, skipping."

echo "=== Step 2: Install 3.9-compatible packages (pinned, --no-deps) ==="
$PIP install --no-deps \
    torch-einops-utils==0.0.30 \
    vit-pytorch==1.18.0 \
    assoc-scan==0.0.4 \
    accelerated-scan==0.2.0 \
    beartype==0.22.2 \
    discrete-continuous-embed-readout==0.2.1 \
    hyper-connections==0.4.9 \
    ema-pytorch==0.7.9 \
    packaging

echo "=== Step 3: Install PoPE-pytorch (bypass >=3.10 requires-python gate) ==="
$PIP install PoPE-pytorch==0.1.4 --no-deps --ignore-requires-python

echo "=== Step 4: Install accelerate (last 3.9-compatible version) ==="
$PIP install accelerate==1.10.0

echo "=== Step 5: Install gym3 + missing minerl/VPT runtime deps ==="
$PIP install --no-deps \
    gym3==0.3.3 \
    cffi==1.17.1 \
    Cython==0.29.37 \
    attrs==26.1.0 \
    platformdirs==4.4.0 \
    pycparser==2.23 \
    ImageIO==2.37.2 \
    imageio-ffmpeg==0.3.0 \
    glfw==1.12.0 \
    glcontext==3.0.0 \
    moderngl==5.12.0 \
    xml2pytorch==0.0.10

echo "=== Step 6: Defensive typing-backport uninstall (crept back via transitive deps) ==="
$PIP uninstall typing -y 2>/dev/null || echo "  Not installed, skipping."

echo "=== Step 7: Patch x_mlps_pytorch/ensemble.py ==="
ENSEMBLE_FILE="${VENV_DIR}/lib/python3.9/site-packages/x_mlps_pytorch/ensemble.py"
if ! head -1 "$ENSEMBLE_FILE" | grep -q "from __future__ import annotations"; then
    sed -i '1s/^/from __future__ import annotations\n/' "$ENSEMBLE_FILE"
    echo "  Patched: added 'from __future__ import annotations'"
else
    echo "  Already patched."
fi

echo "=== Step 8: Patch discrete_continuous_embed_readout ==="
DCER_FILE="${VENV_DIR}/lib/python3.9/site-packages/discrete_continuous_embed_readout/discrete_continuous_embed_readout.py"
if grep -q "^from beartype import beartype$" "$DCER_FILE" 2>/dev/null; then
    # Replace the bare beartype import with a version-gated no-op so Python < 3.10
    # doesn't choke on beartype's PEP 604 use-site type checks.
    sed -i 's|^from beartype import beartype$|import sys\nif sys.version_info >= (3, 10):\n    from beartype import beartype\nelse:\n    def beartype(fn): return fn|' "$DCER_FILE"
    echo "  Patched: beartype no-op on <3.10"
else
    echo "  beartype line already patched or absent."
fi
if grep -q "^from typing import Callable$" "$DCER_FILE" 2>/dev/null; then
    sed -i 's|^from typing import Callable$|from typing import Callable, Union|' "$DCER_FILE"
    echo "  Patched: added Union to typing imports"
fi
if grep -qF "SelectorConfig = tuple[DiscreteConfig, ContinuousConfig] | DiscreteConfig | ContinuousConfig" "$DCER_FILE" 2>/dev/null; then
    # Module-level assignments are not deferred by `from __future__ import annotations`,
    # so PEP 604 `X | Y` syntax here crashes on Python 3.9.
    sed -i 's#^SelectorConfig = tuple\[DiscreteConfig, ContinuousConfig\] | DiscreteConfig | ContinuousConfig$#SelectorConfig = Union[tuple[DiscreteConfig, ContinuousConfig], DiscreteConfig, ContinuousConfig]#' "$DCER_FILE"
    echo "  Patched: SelectorConfig rewritten with Union"
fi

echo "=== Step 9: Verify ==="
$PYTHON -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'Video-Pre-Training')
from dreamer4 import VideoTokenizer, DynamicsWorldModel
from x_mlps_pytorch.ensemble import Ensemble
from PoPE_pytorch import PoPE
from vit_pytorch.vit_with_decorr import DecorrelationLoss
from assoc_scan import AssocScan
from discrete_continuous_embed_readout import MultiCategorical
from hyper_connections import mc_get_init_and_expand_reduce_stream_functions
from torch_einops_utils import maybe
from dreamer4_minecraft_agent import Dreamer4MinecraftAgent
from agent import ENV_KWARGS
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
import minerl, gym, torch, cv2
print('All imports OK -- Python ' + sys.version.split()[0] + ', torch ' + torch.__version__ + ', cuda_available=' + str(torch.cuda.is_available()))
"

echo "=== Done ==="
