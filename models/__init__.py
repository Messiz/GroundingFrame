from .trans_vg import TransVG
from .gloria_grounding import PretrainedGroundingModel


def build_model(args):
    model_name = args.model_name
    if model_name == 'gloria':
        model = PretrainedGroundingModel(args)
    else:
        model = TransVG(args)
    return model

