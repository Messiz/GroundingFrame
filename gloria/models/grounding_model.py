import torch
from torch import nn


class PretrainedGroundingModel(nn.Module):
    def __init__(self,
                 image_encoder: nn.Module,
                 text_encoder: nn.Module,
                 image_dim: int,
                 text_dim: int,
                 freeze_encoder: bool = False
                 ):
        super(PretrainedGroundingModel, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        # self.cross_attn =
        if freeze_encoder:
            for param in self.img_encoder.parameters():
                param.requires_grad = False

    def forword(self, text, image):
       img_emb = self.image_encoder(image)
       text_emb = self.text_encoder(text)
