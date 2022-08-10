from .input import build as build_dvs
from .input_asyn import build as build_asyn


def build_dataset(image_set, args, seq_scene, seq_id):
    return build_dvs(image_set, args, seq_scene, seq_id)


def build_dataset_asyn(image_set, args, seq_scene, seq_id):
    return build_asyn(image_set, args, seq_scene, seq_id)
