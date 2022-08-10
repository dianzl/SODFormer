from .deformable_detr import build_model, build_criterion
from .ST_detr import build_ST
from .Multimodal_ST_detr import build_Multimodal_ST
from .Multimodal_ST_detr_asyn import build_Multimodal_ST as build_Multimodal_ST_asyn


def build_spatial(args):
    return build_model(args)


def build_temporal(args, input_type):
    return build_ST(args, input_type)


def build_fusion(args):
    return build_Multimodal_ST(args)


def build_fusion_asyn(args):
    return build_Multimodal_ST_asyn(args)


def build_cri_pro(args):
    return build_criterion(args)
