import torch
from torch import nn
from transformers import AutoTokenizer

import loss
from gloria_model import text_model, vision_model


class PretrainedGroundingModel(nn.Module):
    def __init__(self, args):
        super(PretrainedGroundingModel, self).__init__()
        self.args = args
        self.text_encoder = text_model.BertEncoder(args)
        self.img_encoder = vision_model.ImageEncoder(args)

        self.local_loss = loss.gloria_loss.local_loss
        self.global_loss = loss.gloria_loss.global_loss
        self.local_loss_weight = self.args.local_loss_weight
        self.global_loss_weight = self.args.global_loss_weight

        self.temp1 = self.args.temp1
        self.temp2 = self.args.temp2
        self.temp3 = self.args.temp3
        self.batch_size = self.args.batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)
        self.ixtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def forword(self, text, image):
        img_emb = self.image_encoder(image)
        text_emb = self.text_encoder(text)
