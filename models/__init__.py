from .trans_vg import TransVG
from gloria import *


def build_model(args, model_name='default'):
    if model_name == 'gloria':
        # model = build_gloria_model(args)
        print(1)
    else:
        model = TransVG(args)
    return model


# def build_gloria_model(args):
#     base_model = gloria.load_gloria()
