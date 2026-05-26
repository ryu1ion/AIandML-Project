from src.students.mobilenetv2 import MobileNetV2Student, make_mobilenetv2_student
from src.students.resnet50 import ResNet50Backbone, make_resnet50


def get_student(name: str, pretrained: bool = False):
    """Factory: return a student backbone exposing forward_features(x)."""
    name = name.lower()
    if name in {"mobilenetv2", "mobilenetv2_100"}:
        return make_mobilenetv2_student(pretrained=pretrained)
    if name in {"resnet50", "r50", "resnet-50"}:
        return make_resnet50(pretrained=pretrained)
    raise ValueError(f"Unknown student: {name}")


__all__ = [
    "MobileNetV2Student",
    "make_mobilenetv2_student",
    "ResNet50Backbone",
    "make_resnet50",
    "get_student",
]
