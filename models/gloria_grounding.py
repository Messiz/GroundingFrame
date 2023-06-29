import argparse
from PIL import Image
import cv2
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn import metrics
import utils.gloria_utils as utils

import loss
from .gloria_model import text_model, vision_model


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(CrossAttention, self).__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        # nn.MultiheadAttention()

        self.query_transform = nn.Linear(query_dim, key_dim)  # 768-->2048
        self.key_transform = nn.Linear(key_dim, key_dim)  # 2048-->2048
        self.value_transform = nn.Linear(value_dim, value_dim)  # 2048-->2048

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        transformed_query = self.query_transform(query)  # (batch_size, query_seq_len, key_dim)
        transformed_key = self.key_transform(key)  # (batch_size, key_seq_len, key_dim)
        transformed_value = self.value_transform(value)  # (batch_size, value_seq_len, value_dim)

        scores = torch.matmul(transformed_query,
                              transformed_key.transpose(1, 2))  # (batch_size, query_seq_len, key_seq_len)
        attention_weights = self.softmax(scores)  # (batch_size, query_seq_len, key_seq_len)

        weighted_sum = torch.matmul(attention_weights, transformed_value)  # (batch_size, query_seq_len, value_dim)

        return weighted_sum


class PretrainedGroundingModel(nn.Module):
    def __init__(self, args):
        super(PretrainedGroundingModel, self).__init__()
        self.cross_attention = None
        self.args = args
        self.text_encoder = text_model.BertEncoder(args)
        self.img_encoder = vision_model.ImageEncoder(args)

        # self.attention = nn.MultiheadAttention()

        self.local_loss = loss.gloria_loss.local_loss
        self.global_loss = loss.gloria_loss.global_loss
        self.local_loss_weight = self.args.local_loss_weight
        self.global_loss_weight = self.args.global_loss_weight

        self.temp1 = self.args.temp1
        self.temp2 = self.args.temp2
        self.temp3 = self.args.temp3
        self.batch_size = self.args.batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.text_bert_type)
        self.ixtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids):
        text_emb_l, text_emb_g, sents = self.text_encoder(
            caption_ids, attention_mask, token_type_ids
        )
        return text_emb_l, text_emb_g, sents

    def image_encoder_forward(self, imgs):
        img_feat_g, img_emb_l = self.img_encoder(imgs, get_local=True)
        img_emb_g, img_emb_l = self.img_encoder.generate_embeddings(
            img_feat_g, img_emb_l
        )

        return img_emb_l, img_emb_g

    def _calc_local_loss(self, img_emb_l, text_emb_l, sents):

        cap_lens = [
            len([w for w in sent if not w.startswith("[")]) + 1 for sent in sents
        ]
        l_loss0, l_loss1, attn_maps = self.local_loss(
            img_emb_l,
            text_emb_l,
            cap_lens,
            temp1=self.temp1,
            temp2=self.temp2,
            temp3=self.temp3,
        )
        return l_loss0, l_loss1, attn_maps

    def _calc_global_loss(self, img_emb_g, text_emb_g):
        g_loss0, g_loss1 = self.global_loss(img_emb_g, text_emb_g, temp3=self.temp3)
        return g_loss0, g_loss1

    def calc_loss(self, img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents):

        l_loss0, l_loss1, attn_maps = self._calc_local_loss(
            img_emb_l, text_emb_l, sents
        )
        g_loss0, g_loss1 = self._calc_global_loss(img_emb_g, text_emb_g)

        # weighted loss
        loss = 0
        loss += (l_loss0 + l_loss1) * self.local_loss_weight
        loss += (g_loss0 + g_loss1) * self.global_loss_weight

        return loss, attn_maps

    def forward(self, image, text):
        self.conv = nn.Conv2d(1024, 768, kernel_size=(1, 1), stride=(1, 1), padding=0)
        img_emb, local_fea = self.img_encoder(image.tensors, get_local=True)
        word_embeddings, sent_embeddings, sents = self.text_encoder(text.tensors, text.mask)
        img_emb_local = self.conv(local_fea)
        # shape:
        #   word_embeddings: [batch, embed_dim, seq_len]    32 x 768 x 20
        #   sent_embeddings: [batch, embed_dim]             32 x 768
        #   sents: [batch, seq_len]                         32 x 20
        #   img_emb: [batch, img_dim]                       32 x 2048
        #   local_fea: [batch, img_dim, h, w]               32 x 1024 x 19 x 19
        #   len(attention_map)                              32
        #   attention_map[0]:                               1 x seq_len(有效长度) x 19 x 19
        #   weight_maps:                                    batch_size x 19 x 19

        attention_maps = self.get_attn_maps(img_emb_local, word_embeddings, sents)
        self.plot_attn_maps(attention_maps, image.tensors, sents)
        weight_maps = torch.cat([attention_map[:, 0, :, :] for attention_map in attention_maps], dim=0)
        # weight_maps = torch.cat([attn_map.mean(dim=1) for attn_map in attention_maps], dim=0)
        # weight_maps = torch.stack(attention_maps).mean(dim=1)
        print(weight_maps.shape)

        # print(attention_maps[0].shape)
        # weight = attention_maps[0].squeeze(0).mean(dim=0)
        # print(weight.shape)
        # # 创建热力图
        # heatmap = plt.imshow(weight, cmap='hot', alpha=0.8)  # 假设 batch=1，取第一张图片的注意力权重
        #
        # # 显示原始图片
        # plt.imshow(image.tensors[0])
        #
        # # 将热力图叠加在原始图片上
        # plt.colorbar(heatmap)
        #
        # print(len(attention_maps))
        # print(attention_maps[0].shape)

        return weight_maps

    def get_global_similarities(self, img_emb_g, text_emb_g):
        img_emb_g = img_emb_g.detach().cpu().numpy()
        text_emb_g = text_emb_g.detach().cpu().numpy()
        global_similarities = metrics.pairwise.cosine_similarity(img_emb_g, text_emb_g)
        global_similarities = torch.Tensor(global_similarities)
        return global_similarities

    def get_local_similarities(self, img_emb_l, text_emb_l, cap_lens):

        batch_size = img_emb_l.shape[0]
        similarities = []

        for i in range(len(text_emb_l)):
            words_num = cap_lens[i]
            word = (
                text_emb_l[i, :, 1: words_num + 1].unsqueeze(0).contiguous()
            )  # [1, 768, 25]

            word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
            context = img_emb_l  # [48, 768, 19, 19]

            weiContext, attn = loss.gloria_loss.attention_fn(
                word, context, 4.0
            )  # [48, 768, 25], [48, 25, 19, 19]

            word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
            weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

            word = word.view(batch_size * words_num, -1)  # [1200, 768]
            weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]
            #
            row_sim = loss.gloria_loss.cosine_similarity(word, weiContext)
            row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

            row_sim.mul_(5.0).exp_()
            row_sim, max_row_idx = torch.max(row_sim, dim=1, keepdim=True)  # [48, 1]

            row_sim = torch.log(row_sim)

            similarities.append(row_sim)

        local_similarities = torch.cat(similarities, 1).detach().cpu()

        return local_similarities

    def get_attn_maps(self, img_emb_l, text_emb_l, sents):
        _, _, attn_maps = self._calc_local_loss(img_emb_l, text_emb_l, sents)
        return attn_maps

    def plot_attn_maps(self, attn_maps, imgs, sents, epoch_idx=0, batch_idx=0):

        img_set, _ = utils.build_attention_images(
            imgs,
            attn_maps,
            max_word_num=self.args.data_text_word_num,
            nvis=self.args.train_nvis,
            rand_vis=self.args.train_rand_vis,
            sentences=sents,
        )

        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = (
                f"{self.args.output_dir}/"
                f"attention_maps_epoch{epoch_idx}_"
                f"{batch_idx}.png"
            )
            im.save(fullpath)

    # def process_text(self, text, device):
    #
    #     if type(text) == str:
    #         text = [text]
    #
    #     processed_text_tensors = []
    #     for t in text:
    #         # use space instead of newline
    #         t = t.replace("\n", " ")
    #
    #         # split sentences
    #         splitter = re.compile("[0-9]+\.")
    #         captions = splitter.split(t)
    #         captions = [point.split(".") for point in captions]
    #         captions = [sent for point in captions for sent in point]
    #
    #         all_sents = []
    #
    #         for t in captions:
    #             t = t.replace("\ufffd\ufffd", " ")
    #             tokenizer = RegexpTokenizer(r"\w+")
    #             tokens = tokenizer.tokenize(t.lower())
    #
    #             if len(tokens) <= 1:
    #                 continue
    #
    #             included_tokens = []
    #             for t in tokens:
    #                 t = t.encode("ascii", "ignore").decode("ascii")
    #                 if len(t) > 0:
    #                     included_tokens.append(t)
    #             all_sents.append(" ".join(included_tokens))
    #
    #         t = " ".join(all_sents)
    #
    #         text_tensors = self.tokenizer(
    #             t,
    #             return_tensors="pt",
    #             truncation=True,
    #             padding="max_length",
    #             max_length=self.cfg.data.text.word_num,
    #         )
    #         text_tensors["sent"] = [
    #             self.ixtoword[ix] for ix in text_tensors["input_ids"][0].tolist()
    #         ]
    #         processed_text_tensors.append(text_tensors)
    #
    #     caption_ids = torch.stack([x["input_ids"] for x in processed_text_tensors])
    #     attention_mask = torch.stack(
    #         [x["attention_mask"] for x in processed_text_tensors]
    #     )
    #     token_type_ids = torch.stack(
    #         [x["token_type_ids"] for x in processed_text_tensors]
    #     )
    #
    #     if len(text) == 1:
    #         caption_ids = caption_ids.squeeze(0).to(device)
    #         attention_mask = attention_mask.squeeze(0).to(device)
    #         token_type_ids = token_type_ids.squeeze(0).to(device)
    #     else:
    #         caption_ids = caption_ids.squeeze().to(device)
    #         attention_mask = attention_mask.squeeze().to(device)
    #         token_type_ids = token_type_ids.squeeze().to(device)
    #
    #     cap_lens = []
    #     for txt in text:
    #         cap_lens.append(len([w for w in txt if not w.startswith("[")]))
    #
    #     return {
    #         "caption_ids": caption_ids,
    #         "attention_mask": attention_mask,
    #         "token_type_ids": token_type_ids,
    #         "cap_lens": cap_lens,
    #     }

    # def process_class_prompts(self, class_prompts, device):
    #
    #     cls_2_processed_txt = {}
    #     for k, v in class_prompts.items():
    #         cls_2_processed_txt[k] = self.process_text(v, device)
    #
    #     return cls_2_processed_txt

    # def process_img(self, paths, device):
    #
    #     transform = builder.build_transformation(self.cfg, split="test")
    #
    #     if type(paths) == str:
    #         paths = [paths]
    #
    #     all_imgs = []
    #     for p in paths:
    #         x = cv2.imread(str(p), 0)
    #
    #         # tranform images
    #         x = self._resize_img(x, self.cfg.data.image.imsize)
    #         img = Image.fromarray(x).convert("RGB")
    #         img = transform(img)
    #         all_imgs.append(torch.tensor(img))
    #
    #     all_imgs = torch.stack(all_imgs).to(device)
    #
    #     return all_imgs

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--pretrained_gloria_model_name', default='gloria_resnet50', type=str)
    parser.add_argument('--local_loss_weight', default=1.0, type=float)
    parser.add_argument('--global_loss_weight', default=1.0, type=float)
    parser.add_argument('--temp1', default=4.0, type=float)
    parser.add_argument('--temp2', default=5.0, type=float)
    parser.add_argument('--temp3', default=10.0, type=float)
    parser.add_argument('--vision_model_name', default='resnet_50', type=str)
    parser.add_argument('--vision_freeze_cnn', default=False, type=bool)
    parser.add_argument('--vision_pretrained', default=True, type=bool)
    parser.add_argument('--text_bert_type', default='emilyalsentzer/Bio_ClinicalBERT', type=str)
    parser.add_argument('--text_last_n_layers', default=4, type=int)
    parser.add_argument('--text_aggregate_method', default='sum', type=str)
    parser.add_argument('--text_norm', default=False, type=bool)
    parser.add_argument('--text_embedding_dim', default=768, type=int)
    parser.add_argument('--text_freeze_bert', default=False, type=bool)
    parser.add_argument('--text_agg_tokens', default=True, type=bool)

    parser.add_argument('--vision_num_targets', default=5, type=int)
    parser.add_argument('--batch_size', default=2, type=int)

    args = parser.parse_args()

    a = np.ones([2, 3, 4, 5])
    a = torch.tensor(a)
    model = PretrainedGroundingModel(args)
    b = model(a, a)
    print(b)

    [['[CLS]', 'bibasilar', 'opacities', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'bibasilar', 'opacities', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'bilateral', 'multifocal', 'areas', 'of', 'consolidation', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'bilateral', 'multifocal', 'areas', 'of', 'consolidation', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'large', 'right', '-', 'sided', 'pneumothorax', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'more', 'dense', 'consolidation', 'at', 'the', 'right', 'lung', 'base', 'raises', 'possibility', 'of',
      'superimposed', 'infection', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'large', 'right', 'pneumothorax', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'left', 'basilar', 'opacity', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'persistent', 'right', 'middle', 'and', 'lower', 'lobe', 'opacities', 'consistent', 'with', 'pneumonia',
      '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'small', 'right', 'pneumothorax', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'left', 'small', '-', 'to', '-', 'moderate', 'apical', 'pneumothorax', '[SEP]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'small', 'right', 'apical', 'pneumothorax', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'small', 'volume', 'of', 'medial', 'pneumothorax', 'or', 'pneumomediastinum', '[SEP]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'focal', 'opacity', 'in', 'the', 'lingula', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'small', 'loculated', 'left', 'pneumothorax', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'left', 'apical', 'pneumothorax', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'bibasilar', 'areas', 'of', 'consolidation', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'bibasilar', 'areas', 'of', 'consolidation', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'air', 'bronchograms', 'extending', 'from', 'the', 'left', 'hilum', 'throughout', 'the', 'left', 'lung',
      'which', 'has', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'right', '-', 'sided', 'basal', 'pneumothorax', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'bilateral', 'basal', 'opacities', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'bilateral', 'basal', 'opacities', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'millimetric', 'right', 'pneumothorax', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'right', 'pneumothorax', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'bibasilar', 'pneumonia', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'bibasilar', 'pneumonia', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'persistent', 'apical', 'lateral', 'pneumothorax', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'bilateral', 'pneumothoraces', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'bilateral', 'pneumothoraces', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'apical', 'component', 'of', 'pneumothorax', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'minimal', 'lateral', 'pneumothorax', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
     ['[CLS]', 'small', 'medial', 'left', '-', 'sided', 'pneumothorax', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]',
      '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']]
