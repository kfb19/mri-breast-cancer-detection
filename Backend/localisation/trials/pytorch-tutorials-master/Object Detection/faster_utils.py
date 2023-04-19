""" DOCSTRING """

import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.patches as patches

import torch
from torchvision import ops

# -------------- Data Untils -------------------


# pylint: disable=E1101
# pylint: disable=E1102
# pylint: disable=W0612
def parse_annotation(annotation_path, image_dir, img_size):
    '''
    Traverse the xml tree, get the annotations, and resize them to
    the scaled image size
    '''
    img_h, img_w = img_size

    with open(annotation_path, "r", encoding="utf-8") as file:
        tree = ET.parse(file)

    root = tree.getroot()

    img_paths = []
    gt_boxes_all = []
    gt_classes_all = []
    # get image paths
    for object_ in root.findall('image'):
        img_path = os.path.join(image_dir, object_.get("name"))
        img_paths.append(img_path)

        # get raw image size
        orig_w = int(object_.get("width"))
        orig_h = int(object_.get("height"))

        # get bboxes and their labels
        groundtruth_boxes = []
        groundtruth_classes = []
        for box_ in object_.findall('box'):
            xmin = float(box_.get("xtl"))
            ymin = float(box_.get("ytl"))
            xmax = float(box_.get("xbr"))
            ymax = float(box_.get("ybr"))

            # rescale bboxes
            bbox = torch.Tensor([xmin, ymin, xmax, ymax])
            bbox[[0, 2]] = bbox[[0, 2]] * img_w/orig_w
            bbox[[1, 3]] = bbox[[1, 3]] * img_h/orig_h

            groundtruth_boxes.append(bbox.tolist())

            # get labels
            label = box_.get("label")
            groundtruth_classes.append(label)

        gt_boxes_all.append(torch.Tensor(groundtruth_boxes))
        gt_classes_all.append(groundtruth_classes)

    return gt_boxes_all, gt_classes_all, img_paths

# -------------- Prepocessing utils ----------------


def calc_gt_offsets(pos_anc_coords, gt_bbox_mapping):
    """ DOCSTRING """
    pos_anc_coords = ops.box_convert(
        pos_anc_coords, in_fmt='xyxy', out_fmt='cxcywh')
    gt_bbox_mapping = ops.box_convert(
        gt_bbox_mapping, in_fmt='xyxy', out_fmt='cxcywh')

    gt_cx, gt_cy, gt_w, gt_h = gt_bbox_mapping[:, 0], \
        gt_bbox_mapping[:, 1], gt_bbox_mapping[:, 2], gt_bbox_mapping[:, 3]
    anc_cx, anc_cy, anc_w, anc_h = pos_anc_coords[:, 0], \
        pos_anc_coords[:, 1], pos_anc_coords[:, 2], pos_anc_coords[:, 3]

    tx_ = (gt_cx - anc_cx)/anc_w
    ty_ = (gt_cy - anc_cy)/anc_h
    tw_ = torch.log(gt_w / anc_w)
    th_ = torch.log(gt_h / anc_h)

    return torch.stack([tx_, ty_, tw_, th_], dim=-1)


def gen_anc_centers(out_size):
    """ DOCSTRING """
    out_h, out_w = out_size

    anc_pts_x = torch.arange(0, out_w) + 0.5
    anc_pts_y = torch.arange(0, out_h) + 0.5

    return anc_pts_x, anc_pts_y


def project_bboxes(bboxes, width_scale_factor,
                   height_scale_factor, mode='a2p'):
    """ DOCSTRING """
    assert mode in ['a2p', 'p2a']

    batch_size = bboxes.size(dim=0)
    proj_bboxes = bboxes.clone().reshape(batch_size, -1, 4)
    invalid_bbox_mask = proj_bboxes == -1  # indicating padded bboxes

    if mode == 'a2p':
        # activation map to pixel image
        proj_bboxes[:, :, [0, 2]] *= width_scale_factor
        proj_bboxes[:, :, [1, 3]] *= height_scale_factor
    else:
        # pixel image to activation map
        proj_bboxes[:, :, [0, 2]] /= width_scale_factor
        proj_bboxes[:, :, [1, 3]] /= height_scale_factor

    # fill padded bboxes back with -1
    proj_bboxes.masked_fill_(invalid_bbox_mask, -1)
    proj_bboxes.resize_as_(bboxes)

    return proj_bboxes


def generate_proposals(anchors, offsets):
    """ DOCSTRING """

    # change format of the anchor boxes from 'xyxy' to 'cxcywh'
    anchors = ops.box_convert(anchors, in_fmt='xyxy', out_fmt='cxcywh')

    # apply offsets to anchors to create proposals
    proposals_ = torch.zeros_like(anchors)
    proposals_[:, 0] = anchors[:, 0] + offsets[:, 0]*anchors[:, 2]
    proposals_[:, 1] = anchors[:, 1] + offsets[:, 1]*anchors[:, 3]
    proposals_[:, 2] = anchors[:, 2] * torch.exp(offsets[:, 2])
    proposals_[:, 3] = anchors[:, 3] * torch.exp(offsets[:, 3])

    # change format of proposals back from 'cxcywh' to 'xyxy'
    proposals = ops.box_convert(proposals_, in_fmt='cxcywh', out_fmt='xyxy')

    return proposals


def gen_anc_base(anc_pts_x, anc_pts_y, anc_scales, anc_ratios, out_size):
    """ DOCSTRING """
    n_anc_boxes = len(anc_scales) * len(anc_ratios)
    anc_base = torch.zeros(1, anc_pts_x.size(dim=0), anc_pts_y.size(
        dim=0), n_anc_boxes, 4)  # shape - [1, Hmap, Wmap, n_anchor_boxes, 4]

    for i_x, x_c in enumerate(anc_pts_x):
        for j_x, y_c in enumerate(anc_pts_y):
            anc_boxes = torch.zeros((n_anc_boxes, 4))
            count = 0
            for i_counter, scale in enumerate(anc_scales):
                for j, ratio in enumerate(anc_ratios):
                    w_param = scale * ratio
                    h_param = scale

                    xmin = x_c - w_param / 2
                    ymin = y_c - h_param / 2
                    xmax = x_c + w_param / 2
                    ymax = y_c + h_param / 2

                    anc_boxes[count, :] = torch.Tensor(
                        [xmin, ymin, xmax, ymax])
                    count += 1

            anc_base[:, i_x, j_x, :] = ops.clip_boxes_to_image(
                anc_boxes, size=out_size)

    return anc_base


def get_iou_mat(batch_size, anc_boxes_all, gt_bboxes_all):
    """ DOCSTRING """

    # flatten anchor boxes
    anc_boxes_flat = anc_boxes_all.reshape(batch_size, -1, 4)
    # get total anchor boxes for a single image
    tot_anc_boxes = anc_boxes_flat.size(dim=1)

    # create a placeholder to compute IoUs amongst the boxes
    ious_mat = torch.zeros(
        (batch_size, tot_anc_boxes, gt_bboxes_all.size(dim=1)))

    # compute IoU of the anc boxes with the gt boxes for all the images
    for i in range(batch_size):
        gt_bboxes = gt_bboxes_all[i]
        anc_boxes = anc_boxes_flat[i]
        ious_mat[i, :] = ops.box_iou(anc_boxes, gt_bboxes)

    return ious_mat


def get_req_anchors(anc_boxes_all, gt_bboxes_all, gt_classes_all,
                    pos_thresh=0.7, neg_thresh=0.2):
    '''
    Prepare necessary data required for training

    Input
    ------
    anc_boxes_all - torch.Tensor of shape:
        (B, w_amap, h_amap, n_anchor_boxes, 4)
        all anchor boxes for a batch of images
    gt_bboxes_all - torch.Tensor of shape (B, max_objects, 4)
        padded ground truth boxes for a batch of images
    gt_classes_all - torch.Tensor of shape (B, max_objects)
        padded ground truth classes for a batch of images

    Returns
    ---------
    positive_anc_ind -  torch.Tensor of shape (n_pos,)
        flattened positive indices for all the images in the batch
    negative_anc_ind - torch.Tensor of shape (n_pos,)
        flattened positive indices for all the images in the batch
    GT_conf_scores - torch.Tensor of shape (n_pos,), IoU scores of +ve anchors
    GT_offsets -  torch.Tensor of shape (n_pos, 4),
        offsets between +ve anchors and their corresponding ground truth boxes
    GT_class_pos - torch.Tensor of shape (n_pos,)
        mapped classes of +ve anchors
    positive_anc_coords - (n_pos, 4) coords of +ve anchors (for visualization)
    negative_anc_coords - (n_pos, 4) coords of -ve anchors (for visualization)
    positive_anc_ind_sep - list of indices to keep track of +ve anchors
    '''
    # get the size and shape parameters
    b_param, w_amap, h_amap, a_param, _ = anc_boxes_all.shape
    max_box_no = gt_bboxes_all.shape[1]
    # max number of groundtruth bboxes in a batch

    # get total number of anchor boxes in a single image
    tot_anc_boxes = a_param * w_amap * h_amap

    # get the iou matrix which contains iou of every anchor box
    # against all the groundtruth bboxes in an image
    iou_mat = get_iou_mat(b_param, anc_boxes_all, gt_bboxes_all)

    # for every groundtruth bbox in an image, find the iou
    # with the anchor box which it overlaps the most
    max_iou_per_gt_box, _ = iou_mat.max(dim=1, keepdim=True)

    # get positive anchor boxes

    # condition 1: the anchor box with the max iou for every gt bbox
    positive_anc_mask = torch.logical_and(
        iou_mat == max_iou_per_gt_box, max_iou_per_gt_box > 0)
    # condition 2: anchor boxes with iou above a threshold with any
    # of the gt bboxes
    positive_anc_mask = torch.logical_or(
        positive_anc_mask, iou_mat > pos_thresh)

    positive_anc_ind_sep = torch.where(positive_anc_mask)[
        0]  # get separate indices in the batch
    # combine all the batches and get the idxs of the +ve anchor boxes
    positive_anc_mask = positive_anc_mask.flatten(start_dim=0, end_dim=1)
    positive_anc_ind = torch.where(positive_anc_mask)[0]

    # for every anchor box, get the iou and the idx of the
    # gt bbox it overlaps with the most
    max_iou_per_anc, max_iou_per_anc_ind = iou_mat.max(dim=-1)
    max_iou_per_anc = max_iou_per_anc.flatten(start_dim=0, end_dim=1)

    # get iou scores of the +ve anchor boxes
    gt_conf_scores = max_iou_per_anc[positive_anc_ind]

    # get gt classes of the +ve anchor boxes

    # expand gt classes to map against every anchor box
    gt_classes_expand = gt_classes_all.view(
        b_param, 1, max_box_no).expand(b_param, tot_anc_boxes, max_box_no)
    # for every anchor box, consider only the class of the gt bbox it
    # overlaps with the most
    gt_class = torch.gather(gt_classes_expand, -1,
                            max_iou_per_anc_ind.unsqueeze(-1)).squeeze(-1)
    # combine all the batches and get the mapped classes of the
    # +ve anchor boxes
    gt_class = gt_class.flatten(start_dim=0, end_dim=1)
    gt_class_pos = gt_class[positive_anc_ind]

    # get gt bbox coordinates of the +ve anchor boxes

    # expand all the gt bboxes to map against every anchor box
    gt_bboxes_expand = gt_bboxes_all.view(
        b_param, 1, max_box_no, 4).expand(b_param, tot_anc_boxes,
                                          max_box_no, 4)
    # for every anchor box, consider only the coordinates of the gt bbox
    # it overlaps with the most
    gt_bboxes = torch.gather(
        gt_bboxes_expand, -2, max_iou_per_anc_ind.reshape(
            b_param, tot_anc_boxes, 1, 1).repeat(1, 1, 1, 4))
    # combine all the batches and get the mapped gt bbox coordinates of
    # the +ve anchor boxes
    gt_bboxes = gt_bboxes.flatten(start_dim=0, end_dim=2)
    gt_bboxes_pos = gt_bboxes[positive_anc_ind]

    # get coordinates of +ve anc boxes
    anc_boxes_flat = anc_boxes_all.flatten(
        start_dim=0, end_dim=-2)  # flatten all the anchor boxes
    positive_anc_coords = anc_boxes_flat[positive_anc_ind]

    # calculate gt offsets
    gt_offsets = calc_gt_offsets(positive_anc_coords, gt_bboxes_pos)

    # get -ve anchors

    # condition: select the anchor boxes with max iou less than the threshold
    negative_anc_mask = max_iou_per_anc < neg_thresh
    negative_anc_ind = torch.where(negative_anc_mask)[0]
    # sample -ve samples to match the +ve samples
    negative_anc_ind = negative_anc_ind[torch.randint(
        0, negative_anc_ind.shape[0], (positive_anc_ind.shape[0],))]
    negative_anc_coords = anc_boxes_flat[negative_anc_ind]

    return positive_anc_ind, negative_anc_ind, gt_conf_scores, gt_offsets, \
        gt_class_pos, positive_anc_coords, negative_anc_coords, \
        positive_anc_ind_sep

# # -------------- Visualization utils ----------------


def display_img(img_data, fig, axes):
    """ DOCSTRING """
    for i, img in enumerate(img_data):
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img)

    return fig, axes


def display_bbox(bboxes, fig, axis, classes=None, in_format='xyxy', color='y',
                 line_width=3):
    """ DOCSTRING """
    if isinstance(bboxes, np.ndarray):
        bboxes = torch.from_numpy(bboxes)
    if classes:
        assert len(bboxes) == len(classes)
    # convert boxes to xywh format
    bboxes = ops.box_convert(bboxes, in_fmt=in_format, out_fmt='xywh')
    c_count = 0
    for box in bboxes:
        x_val, y_val, width, height = box.numpy()
        # display bounding box
        rect = patches.Rectangle(
            (x_val, y_val), width, height, linewidth=line_width,
            edgecolor=color, facecolor='none')
        axis.add_patch(rect)
        # display category
        if classes:
            if classes[c_count] == 'pad':
                continue
            axis.text(x_val + 5, y_val + 20, classes[c_count],
                      bbox=dict(facecolor='yellow', alpha=0.5))
        c_count += 1

    return fig, axis


def display_grid(x_points, y_points, fig, axis, special_point=None):
    """ DOCSTRING """
    # plot grid
    for x_p in x_points:
        for y_p in y_points:
            axis.scatter(x_p, y_p, color="w", marker='+')

    # plot a special point we want to emphasize on the grid
    if special_point:
        x_p, y_p = special_point
        axis.scatter(x_p, y_p, color="red", marker='+')

    return fig, axis
