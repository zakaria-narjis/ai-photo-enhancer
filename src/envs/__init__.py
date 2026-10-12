"""Environment components with lightweight imports."""

from importlib import import_module

from .new_edit_photo import SLIDER_ORDER, PhotoEditor

__all__ = [
    "Action_Space",
    "FiveKDataset",
    "Observation_Space",
    "PhotoEditor",
    "PhotoEnhancementEnv",
    "PhotoEnhancementEnvTest",
    "SLIDER_ORDER",
    "create_dataloaders",
    "dehaze",
]

_LAZY_IMPORTS = {
    "Action_Space": (".photo_env", "Action_Space"),
    "FiveKDataset": (".image_dataset", "FiveKDataset"),
    "Observation_Space": (".photo_env", "Observation_Space"),
    "PhotoEnhancementEnv": (".photo_env", "PhotoEnhancementEnv"),
    "PhotoEnhancementEnvTest": (".photo_env", "PhotoEnhancementEnvTest"),
    "create_dataloaders": (".env_dataloader", "create_dataloaders"),
    "dehaze": (".dehaze.src", "dehaze"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_IMPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    return getattr(import_module(module_name, __name__), attribute)
