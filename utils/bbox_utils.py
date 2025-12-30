def normalize_coordinates(box, image_width, image_height):
    """
    Normalize the coordinates of a single bbox or a list of bboxes to [0,1] range.
    If the box is already normalized, return the original box.

    Args:
        box: A single bbox [x1,y1,x2,y2] or list of bboxes [[x1,y1,x2,y2], ...]
        image_width: The width of the image
        image_height: The height of the image

    Returns:
        Normalized bbox(es) in [0,1] range
    """
    x1, y1, x2, y2 = box
    normalized_box = [
        round((x1 / image_width) * 1000),
        round((y1 / image_height) * 1000),
        round((x2 / image_width) * 1000),
        round((y2 / image_height) * 1000),
    ]
    return normalized_box


def denormalize_coordinates(box, image_width, image_height):
    """
    Denormalize the coordinates of a single bbox or a list of bboxes from [0,1] range to original range.

    Args:
        box: A single bbox [x1,y1,x2,y2] or list of bboxes [[x1,y1,x2,y2], ...]
        image_width: The width of the image
        image_height: The height of the image

    Returns:
        Denormalized bbox(es) in original range
    """
    # print(f"box: {box}, image_width: {image_width}, image_height: {image_height}")
    x1, y1, x2, y2 = box
    denormalized_box = [
        round((x1 / 1000) * image_width),
        round((y1 / 1000) * image_height),
        round((x2 / 1000) * image_width),
        round((y2 / 1000) * image_height),
    ]
    return denormalized_box


def convert_xywh_to_x1y1x2y2(bboxes):
    """Convert bboxes from [x,y,w,h] format to [x1,y1,x2,y2] format.

    Args:
        bboxes: A single bbox [x,y,w,h] or list of bboxes [[x,y,w,h], ...]

    Returns:
        Converted bbox(es) in [x1,y1,x2,y2] format
    """
    if not isinstance(bboxes[0], (list, tuple)):
        # Single bbox
        x1, y1, w, h = bboxes
        return [x1, y1, x1 + w, y1 + h]
    else:
        # Multiple bboxes
        return [[x, y, x + w, y + h] for x, y, w, h in bboxes]


def convert_x1y1x2y2_to_xywh(bboxes):
    """Convert bboxes from [x1,y1,x2,y2] format to [x,y,w,h] format.

    Args:
        bboxes: A single bbox [x1,y1,x2,y2] or list of bboxes [[x1,y1,x2,y2], ...]

    Returns:
        Converted bbox(es) in [x,y,w,h] format
    """
    if not isinstance(bboxes[0], (list, tuple)):
        # Single bbox
        x1, y1, x2, y2 = bboxes
        return [x1, y1, x2 - x1, y2 - y1]
    else:
        # Multiple bboxes
        return [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in bboxes]
