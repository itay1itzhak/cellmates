from joblib import Memory
from tdm.paths import package_root


def _init_peristent_cache():
    memory = Memory(package_root / ".persistent_cache", verbose=-1)
    return memory.cache


def _clear_cache():
    memory = Memory(package_root / ".persistent_cache", verbose=-1)
    memory.clear(warn=True)


persistent_cache = _init_peristent_cache()
