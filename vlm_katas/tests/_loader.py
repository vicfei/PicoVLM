import importlib
import os


def load(name: str):
    package = os.getenv("VLM_IMPL", "exercises")
    if package not in {"exercises", "solutions"}:
        raise RuntimeError("VLM_IMPL 只能是 exercises 或 solutions")
    return importlib.import_module(f"{package}.{name}")
