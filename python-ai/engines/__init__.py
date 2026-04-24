from importlib import import_module

from utils.engine import CANONICAL_ENGINE, normalize_engine_name

ENGINE_REGISTRY = {
    CANONICAL_ENGINE: ("engines.opencv_engine", "MediaPipeEngine"),
}
ENGINE_CACHE = {}


def get_engine(name: str):
    normalized_name = normalize_engine_name(name)
    engine_config = ENGINE_REGISTRY.get(normalized_name)

    module_name, class_name = engine_config

    if normalized_name not in ENGINE_CACHE:
        module = import_module(module_name)
        engine_class = getattr(module, class_name)
        ENGINE_CACHE[normalized_name] = engine_class()

    return ENGINE_CACHE[normalized_name]
