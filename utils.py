import time
import math

import cv2
import numpy as np
import torch
import torchvision
import shapely
import shapely.geometry


def gen_shape(label, points, shape_type):
    shape = {
            "label": label,
            "points": points,
            "group_id": None,
            "shape_type": shape_type,
            "flags": {}
            }
    return shape

def gen_rectangle(label, points):
    return gen_shape(label, points, 'rectangle')

def gen_polygon(label, points):
    return gen_shape(label, points, 'polygon')

def gen_data(image_name, height, width, flags, shapes):
    data = {
    'version': '5.0.1',
    'imageHeight': height,
    'imageWidth': width,
    'imagePath': image_name,
    'imageData': None,
    'flags': flags,
    'shapes': shapes
    }
    return data

def adjust_image_channel(image, channel):
    assert channel == 1 or channel == 3, "Channel must be 1 or 3."
    assert len(image.shape) == 2 or image.shape[2] == 3, "Image's channel must be 1 or 3."

    if len(image.shape) == 2 and channel == 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and channel == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if len(image.shape) == 2:
        image = image[..., None]
    return image

def image_to_tensor(image, mean, std, flip=True):
    tensor = (image - mean) / std
    if flip:
        tensor = tensor.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    else:
        tensor = tensor.transpose((2, 0, 1))
    tensor = tensor[None]
    tensor = np.ascontiguousarray(tensor, dtype=np.float32)
    return tensor

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

# def xywh2xyxy(x):
#     # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
#     y = np.copy(x)
#     y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
#     y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
#     y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
#     y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
#     return y

# def box_iou(box1, box2):
#     lt = np.maximum(box1[:2], box2[:2])
#     rb = np.minimum(box1[2:4], box2[2:4])

#     box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])


#     if lt[0] > rb[0] or lt[1] > rb[1]:
#         return 0.0

#     interBoxS = (rb[0] - lt[0]) * (rb[1] - lt[1])
#     return interBoxS / (box1_area + box2_area - interBoxS)


# def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=300):
#     output = []
#     max_nms = 30000

#     prediction[:, 5:] *= prediction[:, 4:5]

#     score_best = np.max(prediction[:, 5:], axis=1)
#     xc = score_best > conf_thres
#     x = prediction[xc]

#     n = x.shape[0]  # number of boxes
#     if not n:  # no boxes
#         return output

#     score_best = score_best[xc]
#     x = x[np.argsort(-score_best)]
#     arg_best = np.argmax(x[:, 5:], axis=1)

#     if n > max_nms:  # excess boxes
#         x = x[:max_nms]  # sort by confidence

#     # Batched NMS
#     boxes = xywh2xyxy(x[:, :4]).tolist()
#     class_best = arg_best.tolist()

#     while boxes:
#         box1 = boxes.pop(0)
#         cl = class_best.pop(0)
#         output.append((box1, cl))

#         pop_indexes = []
#         for index, box2 in enumerate(boxes):
#             if box_iou(box1, box2) > iou_thres:
#                 pop_indexes.append(index)
#         for index in pop_indexes[::-1]:
#             boxes.pop(index)
#             class_best.pop(index)

#     return output

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)


def non_max_suppression(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        classes=None,
                        agnostic=False,
                        multi_label=False,
                        labels=(),
                        max_det=300):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.3 + 0.03 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            #LOGGER.warning(f'WARNING: NMS time limit {time_limit:.3f}s exceeded') TODO：
            break  # time limit exceeded

    return output

def polygon_inter_union_cpu(boxes1, boxes2):
    """
        Reference: https://github.com/ming71/yolov3-polygon/blob/master/utils/utils.py ;
        iou computation (polygon) with cpu;
        Boxes have shape nx8 and Anchors have mx8;
        Return intersection and union of boxes[i, :] and anchors[j, :] with shape of (n, m).
    """
    
    n, m = boxes1.shape[0], boxes2.shape[0]
    inter = torch.zeros(n, m)
    union = torch.zeros(n, m)
    for i in range(n):
        polygon1 = shapely.geometry.Polygon(boxes1[i, :].view(4,2)).convex_hull
        for j in range(m):
            polygon2 = shapely.geometry.Polygon(boxes2[j, :].view(4,2)).convex_hull
            if polygon1.intersects(polygon2):
                try:
                    inter[i, j] = polygon1.intersection(polygon2).area
                    union[i, j] = polygon1.union(polygon2).area
                except shapely.geos.TopologicalError:
                    print('shapely.geos.TopologicalError occured')
    return inter, union

def polygon_box_iou(boxes1, boxes2, GIoU=False, DIoU=False, CIoU=False, eps=1e-7, device="cpu"):
    """
        Compute iou of polygon boxes via cpu or cuda;
        For cuda code, please refer to files in ./iou_cuda
        Returns the IoU of shape (n, m) between boxes1 and boxes2. boxes1 is nx8, boxes2 is mx8
    """
    
    boxes1, boxes2 = boxes1.clone().to(device), boxes2.clone().to(device)
    if torch.cuda.is_available() and polygon_inter_union_cuda_enable and boxes1.is_cuda:
        # using cuda extension to compute
        # the boxes1 and boxes2 go inside polygon_inter_union_cuda must be torch.cuda.float, not double type
        boxes1_ = boxes1.float().contiguous().view(-1)
        boxes2_ = boxes2.float().contiguous().view(-1)
        inter, union = polygon_inter_union_cuda(boxes2_, boxes1_)  # Careful that order should be: boxes2_, boxes1_.
    else:
        # using shapely (cpu) to compute
        inter, union = polygon_inter_union_cpu(boxes1, boxes2)
    union += eps
    iou = inter / union
    iou[torch.isnan(inter)] = 0.0
    iou[torch.logical_and(torch.isnan(inter), torch.isnan(union))] = 1.0
    iou[torch.isnan(iou)] = 0.0
    
    if GIoU or DIoU or CIoU:
        # minimum bounding box of boxes1 and boxes2
        b1_x1, b1_x2 = boxes1[:, 0::2].min(dim=1)[0], boxes1[:, 0::2].max(dim=1)[0] # 1xn
        b1_y1, b1_y2 = boxes1[:, 1::2].min(dim=1)[0], boxes1[:, 1::2].max(dim=1)[0] # 1xn
        b2_x1, b2_x2 = boxes2[:, 0::2].min(dim=1)[0], boxes2[:, 0::2].max(dim=1)[0] # 1xm
        b2_y1, b2_y2 = boxes2[:, 1::2].min(dim=1)[0], boxes2[:, 1::2].max(dim=1)[0] # 1xm
        for i in range(boxes1.shape[0]):
            cw = torch.max(b1_x2[i], b2_x2) - torch.min(b1_x1[i], b2_x1)  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2[i], b2_y2) - torch.min(b1_y1[i], b2_y1)  # convex height
            if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1[i] - b1_x2[i]) ** 2 +
                        (b2_y1 + b2_y2 - b1_y1[i] - b1_y2[i]) ** 2) / 4  # center distance squared
                if DIoU:
                    iou[i, :] -= rho2 / c2  # DIoU
                elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    w2, h2 = b2_x2-b2_x1, b2_y2-b2_y1+eps
                    w1, h1 = b1_x2[i]-b1_x1[i], b1_y2[i]-b1_y1[i]+eps
                    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou[i, :] + (1 + eps))
                    iou[i, :] -= (rho2 / c2 + v * alpha)  # CIoU
            else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
                c_area = cw * ch + eps  # convex area
                iou[i, :] -= (c_area - union[i, :]) / c_area  # GIoU
    return iou  # IoU

def polygon_nms_kernel(x, iou_thres):
    """
        non maximum suppression kernel for polygon-enabled boxes
        x is the prediction with boxes x[:, :8], confidence x[:, 8], class x[:, 9] 
        Return the selected indices
    """
    
    unique_labels = x[:, 9].unique()
    _, scores_sort_index = torch.sort(x[:, 8], descending=True)
    x = x[scores_sort_index]
    indices = scores_sort_index
    selected_indices = []
    
    # Iterate through all predicted classes
    for unique_label in unique_labels:
        x_ = x[x[:, 9]==unique_label]
        indices_ = indices[x[:, 9]==unique_label]
        
        while x_.shape[0]:
            # Save the indice with the highest confidence
            selected_indices.append(indices_[0])
            if len(x_) == 1: break
            # Compute the IOUs for all other the polygon boxes
            iou = polygon_box_iou(x_[0:1, :8], x_[1:, :8], device=x.device).view(-1)
            # Remove overlapping detections with IoU >= NMS threshold
            x_ = x_[1:][iou < iou_thres]
            indices_ = indices_[1:][iou < iou_thres]
            
    return torch.LongTensor(selected_indices)

def xywhrm2xyxyxyxy(xywhrm):
    """
        xywhrm : shape (N, 6)
        Transform x,y,w,h,re,im to x1,y1,x2,y2,x3,y3,x4,y4
        Suitable for both pixel-level and normalized
    """
    is_array = isinstance(xywhrm, np.ndarray)
    if is_array:
        xywhrm = torch.from_numpy(xywhrm)
        
    x0, x1, y0, y1 = -xywhrm[:, 2:3]/2, xywhrm[:, 2:3]/2, -xywhrm[:, 3:4]/2, xywhrm[:, 3:4]/2
    xyxyxyxy = torch.cat((x0, y0, x1, y0, x1, y1, x0, y1), dim=-1).view(-1, 4, 2).contiguous()
    R = torch.zeros((xyxyxyxy.shape[0], 2, 2), dtype=xyxyxyxy.dtype, device=xyxyxyxy.device)
    R[:, 0, 0], R[:, 1, 1] = xywhrm[:, 4], xywhrm[:, 4]
    R[:, 0, 1], R[:, 1, 0] = xywhrm[:, 5], -xywhrm[:, 5]
    
    xyxyxyxy = torch.matmul(xyxyxyxy, R).view(-1, 8).contiguous()+xywhrm[:, [0, 1, 0, 1, 0, 1, 0, 1]]
    return xyxyxyxy.cpu().numpy() if is_array else xyxyxyxy

def rotate_box_iou(boxes1, boxes2, GIoU=False, DIoU=False, CIoU=False, eps=1e-7, device="cpu"):
    """
        Compute iou of rotated boxes via cpu or cuda;
        For cuda code, please refer to files in ./iou_cuda
        Returns the IoU of shape (n, m) between boxes1 and boxes2. boxes1 is nx6, boxes2 is mx6
    """

    boxes1_xyxyxyxy = xywhrm2xyxyxyxy(boxes1)
    boxes2_xyxyxyxy = xywhrm2xyxyxyxy(boxes2)
    return polygon_box_iou(boxes1_xyxyxyxy, boxes2_xyxyxyxy, GIoU, DIoU, CIoU, eps, device)  # IoU

def rotate_non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """
        Runs Non-Maximum Suppression (NMS) on inference results for rotated boxes
        Returns:  list of detections, on (n,8) tensor per image [x, y, w, h, re, im, conf, cls]
    """
    
    # prediction has the shape of (bs, all potential anchors, 87)
    assert not agnostic, "rotated boxes does not support agnostic"
    nc = prediction.shape[2] - 7  # number of classes
    xc = prediction[..., 6] > conf_thres  # confidence candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 3, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into polygon_nms_kernel, can increase this value
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 8), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 6] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 7), device=x.device)
            v[:, :6] = l[:, 1:7]  # box
            v[:, 6] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 7] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 7:] *= x[:, 6:7]  # conf = obj_conf * cls_conf

        # Box (x, y, w, h, re, im)
        box = x[:, :6].clone()

        # Detections matrix nx8 (xywhrm, conf, cls)
        # Transfer sigmoid probabilities of classes (e.g. three classes [0.567, 0.907, 0.01]) to selected classes (1.0)
        if multi_label:
            i, j = (x[:, 7:] > conf_thres).nonzero(as_tuple=False).T
            # concat satisfied boxes (multi-label-enabled) along 0 dimension
            x = torch.cat((box[i], x[i, j + 7, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 7:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 7:8] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 6].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Rotate NMS does not support Batch NMS and Agnostic
        # x is the sorted predictions with boxes x[:, :6], confidence x[:, 6], class x[:, 7]
        # x_ is the sorted predictions with boxes x_[:, :8], confidence x_[:, 8], class x_[:, 9]
        # cannot use torchvision.ops.nms, which only deals with axis-aligned boxes
        x_ = torch.zeros((x.shape[0], 10), dtype=x.dtype, device=x.device)
        x_[:, 8:10] = x[:, 6:8]
        x_[:, :8] = xywhrm2xyxyxyxy(x[:, :6])
        i = polygon_nms_kernel(x_, iou_thres)  # polygon-NMS

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            boxes = x[:, :6]
            # update boxes as boxes(i,6) = weights(i,n) * rotated boxes(n,6)
            iou = rotate_box_iou(boxes[i], boxes, device=prediction.device) > iou_thres  # iou matrix
            weights = iou * x[:, 6][None]  # rotated box weights
            x[i, :6] = torch.mm(weights, x[:, :6]).float() / weights.sum(1, keepdim=True)  # merged rotated boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
    return output

def rotate_scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (x, y, w, h, re, im) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, 0] -= pad[0]  # x padding
    coords[:, 1] -= pad[1]  # y padding
    coords[:, :4] /= gain
    
    coords[:, 0].clamp_(0, img0_shape[1])  # cx
    coords[:, 1].clamp_(0, img0_shape[0])  # cy
    coords[:, 2].clamp_(0, img0_shape[1])  # width
    coords[:, 3].clamp_(0, img0_shape[0])  # height
    return coords