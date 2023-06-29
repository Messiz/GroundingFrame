import torch
import numpy as np
import torch.nn as nn

from utils.box_utils import bbox_iou, xywh2xyxy


def trans_vg_eval_val(pred_boxes, gt_boxes):
    batch_size = pred_boxes.shape[0]
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu = torch.sum(iou >= 0.5) / float(batch_size)

    return iou, accu


def trans_vg_eval_test(pred_boxes, gt_boxes):
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu_num = torch.sum(iou >= 0.5)

    return accu_num


def eval_pointing_game(attn_maps, gt_boxes):
    # attn_maps shape: [batch_size, 19, 19]
    total_num = attn_maps.shape[0]
    # h, w = attn_maps.shape[1], attn_maps.shape[2]
    h, w = 299, 299
    gt_boxes = xywh2xyxy(gt_boxes)  # x1, y1, x2, y2    (0, 1)之间
    # [batch, 4]
    gt_boxes = gt_boxes * h   # 尺度扩大为299*299
    attn_maps = nn.functional.interpolate(attn_maps.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)
    # attn_maps = nn.functional.interpolate(attn_maps.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)

    max_indices = torch.argmax(attn_maps.view(total_num, -1), dim=1)
    # 将一维的最大值索引转换为二维坐标
    max_row_indices = max_indices // h    # list [batch_size]
    max_col_indices = max_indices % h

    x1 = gt_boxes[:, 0]
    y1 = gt_boxes[:, 1]
    x2 = gt_boxes[:, 2]
    y2 = gt_boxes[:, 3]

    num_hit = 0
    for i in range(total_num):
        x, y = max_col_indices[i], max_row_indices[i]
        if x >= x1[i] and x <= x2[i] and y >= y1[i] and y <= y2[i]:
            num_hit += 1
    print(num_hit)
    return num_hit


