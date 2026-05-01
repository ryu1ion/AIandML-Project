from src.students.mobilenetv2 import MobileNetV2Student, make_mobilenetv2_student


def get_student(name: str, pretrained: bool = False):
    """Factory: return a student backbone exposing forward_features(x)."""
    name = name.lower()
    if name in {"mobilenetv2", "mobilenetv2_100"}:
        return make_mobilenetv2_student(pretrained=pretrained)
    raise ValueError(f"Unknown student: {name}")


__all__ = ["MobileNetV2Student", "make_mobilenetv2_student", "get_student"]
