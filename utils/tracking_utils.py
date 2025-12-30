import math
import cv2 as cv

from utils.bbox_utils import convert_x1y1x2y2_to_xywh


def sample_target(im_path, target_bb, search_area_factor):
    """Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im_path - cv image
        target_bb - target box [x1, y1, x2, y2]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    im = cv.imread(im_path)
    # BGR to RGB
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    x1, y1, x2, y2 = target_bb
    x, y, w, h = convert_x1y1x2y2_to_xywh(target_bb)
    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception("Too small bounding box.")

    x1 = int(round(x + 0.5 * w - crop_sz * 0.5))
    x2 = int(x1 + crop_sz)

    y1 = int(round(y + 0.5 * h - crop_sz * 0.5))
    y2 = int(y1 + crop_sz)

    x1_pad = int(max(0, -x1))
    x2_pad = int(max(x2 - im.shape[1] + 1, 0))

    y1_pad = int(max(0, -y1))
    y2_pad = int(max(y2 - im.shape[0] + 1, 0))

    # Crop target
    im_crop = im[y1 + y1_pad : y2 - y2_pad, x1 + x1_pad : x2 - x2_pad, :]

    return im_crop


# def sample_target(im_path, target_bb, search_area_factor, output_sz=None, mask=None, return_bbox=False):
#     """Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

#     args:
#         im_path - cv image
#         target_bb - target box [x1, y1, x2, y2]
#         search_area_factor - Ratio of crop size to target size
#         output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

#     returns:
#         cv image - extracted crop
#         float - the factor by which the crop has been resized to make the crop size equal output_size
#     """
#     im = cv.imread(im_path)
#     x1, y1, x2, y2 = target_bb
#     x, y, w, h = convert_x1y1x2y2_to_xywh(target_bb)
#     # Crop image
#     crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

#     if crop_sz < 1:
#         raise Exception("Too small bounding box.")

#     x1 = int(round(x + 0.5 * w - crop_sz * 0.5))
#     x2 = int(x1 + crop_sz)

#     y1 = int(round(y + 0.5 * h - crop_sz * 0.5))
#     y2 = int(y1 + crop_sz)

#     x1_pad = int(max(0, -x1))
#     x2_pad = int(max(x2 - im.shape[1] + 1, 0))

#     y1_pad = int(max(0, -y1))
#     y2_pad = int(max(y2 - im.shape[0] + 1, 0))

#     # Crop target
#     im_crop = im[y1 + y1_pad : y2 - y2_pad, x1 + x1_pad : x2 - x2_pad, :]
#     if mask is not None:
#         mask_crop = mask[y1 + y1_pad : y2 - y2_pad, x1 + x1_pad : x2 - x2_pad]

#     # Pad
#     im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
#     # Deal with attention mask
#     H, W, _ = im_crop_padded.shape
#     att_mask = np.ones((H, W))
#     end_x, end_y = -x2_pad, -y2_pad
#     if y2_pad == 0:
#         end_y = None
#     if x2_pad == 0:
#         end_x = None
#     att_mask[y1_pad:end_y, x1_pad:end_x] = 0
#     if mask is not None:
#         mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode="constant", value=0)

#     bbox = torch.tensor([[[0.5 - w / crop_sz / 2, 0.5 - h / crop_sz / 2, w / crop_sz, h / crop_sz]]])
#     if return_bbox:
#         if output_sz is not None:
#             resize_factor = output_sz / crop_sz
#             im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
#             att_mask = cv.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
#             if mask is None:
#                 return im_crop_padded, resize_factor, att_mask, bbox
#             mask_crop_padded = F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode="bilinear", align_corners=False)[0, 0]
#             return im_crop_padded, resize_factor, att_mask, mask_crop_padded, bbox

#         else:
#             if mask is None:
#                 return im_crop_padded, att_mask.astype(np.bool_), 1.0, bbox
#             return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded, bbox
#     else:
#         if output_sz is not None:
#             resize_factor = output_sz / crop_sz
#             im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
#             att_mask = cv.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
#             if mask is None:
#                 return im_crop_padded, resize_factor, att_mask
#             mask_crop_padded = F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode="bilinear", align_corners=False)[0, 0]
#             return im_crop_padded, resize_factor, att_mask, mask_crop_padded

#         else:
#             if mask is None:
#                 return im_crop_padded, att_mask.astype(np.bool_), 1.0
#             return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded
