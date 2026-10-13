"""Relocate the existing conda environment without requiring its old package cache."""
from pathlib import Path
import conda_pack

root = Path('/home/gaob/Dual-RGBT')
conda_pack.pack(prefix='/home/zhouy/miniconda3/envs/pytracking',
                output=str(root / 'pytracking_relocatable.tar.gz'))
