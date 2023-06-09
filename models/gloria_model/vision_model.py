from numpy.lib.function_base import extract
import torch
import torch.nn as nn

from . import cnn_backbones
from omegaconf import OmegaConf


class ImageEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()

        self.output_dim = args.text_embedding_dim

        model_function = getattr(cnn_backbones, args.vision_model_name)
        self.model, self.feature_dim, self.interm_feature_dim = model_function(
            pretrained=args.vision_pretrained
        )

        self.global_embedder = nn.Linear(self.feature_dim, self.output_dim)
        self.local_embedder = nn.Conv2d(
            self.interm_feature_dim,
            self.output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        if args.vision_freeze_cnn:
            print("Freezing CNN model")
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, get_local=False):
        # --> fixed-size input: batch x 3 x 299 x 299
        if "resnet" or "resnext" in self.args.vision_model_name:
            global_ft, local_ft = self.resnet_forward(x, extract_features=True)
        elif "densenet" in self.args.vision_model_name:
            global_ft, local_ft = self.dense_forward(x, extract_features=True)

        if get_local:
            return global_ft, local_ft
        else:
            return global_ft

    def generate_embeddings(self, global_features, local_features):

        global_emb = self.global_embedder(global_features)
        local_emb = self.local_embedder(local_features)

        return global_emb, local_emb

    def resnet_forward(self, x, extract_features=False):

        # --> fixed-size input: batch x 3 x 299 x 299
        print('上采样前：', x.shape)
        x = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True)(x)
        print('上采样后：', x.shape)
        x = self.model.conv1(x)  # (batch_size, 64, 150, 150)
        # print("x: ", x.shape)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        # print("x: ", x.shape)

        x = self.model.layer1(x)  # (batch_size, 64, 75, 75)--(batch_size, 256, 75, 75)
        # print("x: ", x.shape)
        x = self.model.layer2(x)  # (batch_size, 128, 38, 38)--(batch_size, 512, 38, 38)
        # print("x: ", x.shape)
        x = self.model.layer3(x)  # (batch_size, 256, 19, 19)--(batch_size, 1024, 19, 19)
        # print("x: ", x.shape)
        local_features = x
        x = self.model.layer4(x)  # (batch_size, 512, 10, 10)--(batch_size, 2048, 10, 10)

        # print("x: ", x.shape)
        x = self.pool(x)
        # print("x: ", x.shape)
        x = x.view(x.size(0), -1)
        # print("x: ", x.shape)
        return x, local_features  # (batch_size, 2048)

    def densenet_forward(self, x, extract_features=False):
        pass

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)


class PretrainedImageClassifier(nn.Module):
    def __init__(
            self,
            image_encoder: nn.Module,
            num_cls: int,
            feature_dim: int,
            freeze_encoder: bool = True,
    ):
        super(PretrainedImageClassifier, self).__init__()
        self.img_encoder = image_encoder
        self.classifier = nn.Linear(feature_dim, num_cls)
        if freeze_encoder:
            for param in self.img_encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.img_encoder(x)
        pred = self.classifier(x)
        return pred


class ImageClassifier(nn.Module):
    def __init__(self, args, image_encoder=None):
        super(ImageClassifier, self).__init__()

        model_function = getattr(cnn_backbones, args.vision_model_name)
        self.img_encoder, self.feature_dim, _ = model_function(
            pretrained=args.vision_pretrained
        )

        self.classifier = nn.Linear(self.feature_dim, args.vision_num_targets)

    def forward(self, x):
        x = self.img_encoder(x)
        pred = self.classifier(x)
        return pred
