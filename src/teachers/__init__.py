from src.teachers.dino import DinoTeacher, load_dino_vits16


def get_teacher(name: str):
    """Factory: return a frozen, eval-mode teacher exposing forward_features(x)."""
    name = name.lower()
    if name in {"dino_vits16", "dino-vits16", "dino"}:
        return load_dino_vits16()
    raise ValueError(f"Unknown teacher: {name}")


__all__ = [
    "DinoTeacher",
    "load_dino_vits16",
    "get_teacher",
]
