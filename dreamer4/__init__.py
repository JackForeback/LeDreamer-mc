from dreamer4.dreamer4 import (
    VideoTokenizer,
    DynamicsWorldModel,
    AxialSpaceTimeTransformer,
    Actions,
    Experience,
)


def __getattr__(name):
    """Lazy-load trainers to avoid importing accelerate at module load time.

    The evaluation agent only needs VideoTokenizer and DynamicsWorldModel.
    Deferring the trainer import prevents requiring accelerate/ema_pytorch
    in environments where only inference is needed (e.g. the Python 3.9
    check venv used for MineRL evaluation).
    """
    _trainer_names = {'VideoTokenizerTrainer', 'BehaviorCloneTrainer', 'DreamTrainer', 'SimTrainer'}
    if name in _trainer_names:
        from dreamer4 import trainers
        return getattr(trainers, name)
    raise AttributeError(f"module 'dreamer4' has no attribute {name!r}")
