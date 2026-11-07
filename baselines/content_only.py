"""Content-only DeBERTa baseline: the same model and training as CPMG, memory text only.

It is trained by the shared trainer with `configs/content_only.yaml` (variant=content_only):
    python train.py --config configs/content_only.yaml
This module just runs that command so the baseline can also be launched as `python -m baselines.content_only`.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, os.path.join(ROOT, "train.py"), "--config",
                              os.path.join(ROOT, "configs/content_only.yaml")], cwd=ROOT))
