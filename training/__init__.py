"""Hinglish fine-tuning helpers, layered on pocket-tts's own `training` package.

This directory deliberately shares its package name with the official trainer
(`$POCKET_TTS_REPO/training`: train.py, args.py, modules/, dataloader/, ...). The package path is
extended with that checkout, so our modules (train_hinglish, warm_start, param_groups, ...) resolve
here and every official module (training.train, training.modules.builders, ...) resolves there,
unchanged. None of our module names collide with official ones.
"""

import os
from pathlib import Path

_OFFICIAL = Path(os.environ.get("POCKET_TTS_REPO", Path.home() / "vaani_pocket_tts_work" / "pocket-tts")) / "training"
if _OFFICIAL.is_dir() and str(_OFFICIAL) not in __path__:
    __path__.append(str(_OFFICIAL))
