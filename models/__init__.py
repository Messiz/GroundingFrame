from .trans_vg import TransVG
from .gloria_grounding import PretrainedGroundingModel
import torch

_MODELS = {
    "gloria_resnet50": "gloria/pretrained/chexpert_resnet50.ckpt",
    "gloria_resnet18": "gloria/pretrained/chexpert_resnet18.ckpt",
}

def build_model(args):
    model_name = args.model_name
    if model_name == 'gloria':
        model = PretrainedGroundingModel(args)
        # 载入预训练权重
        ckpt_path = _MODELS[args.pretrained_gloria_model_name]
        ckpt = torch.load(ckpt_path)
        # cfg = ckpt["hyper_parameters"]
        ckpt_dict = ckpt["state_dict"]

        fixed_ckpt_dict = {}
        for k, v in ckpt_dict.items():
            new_key = k.split("gloria.")[-1]
            fixed_ckpt_dict[new_key] = v
        ckpt_dict = fixed_ckpt_dict

        model.load_state_dict(ckpt_dict)
    else:
        model = TransVG(args)
    return model

