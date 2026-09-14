"""The two-image prompt used by the released tracking model."""

import json


def build_tracking_prompt(language, color="blue"):
    return (
        "You are a visual object tracker.\n"
        "Given the visual information of the target object, including an initial visual template "
        "<template><image></template>,\n"
        "and the language description of the target object, including an initial language description "
        f"<ref> {json.dumps(language, ensure_ascii=False)} </ref>.\n\n"
        "Then, detect the target object is in search image <search><image></search>.\n"
        "    1. determine whether the target is visible in the search image.\n"
        "    2. return the bounding box of its current position in the search image in the format "
        "[x_min, y_min, x_max, y_max].\n\n"
        "Please return the answer in JSON format, for example: ```json\n{{\n"
        '  "visible": "yes/no", \n  "bbox":[x_min, y_min, x_max, y_max]\n}}\n```.\n\n'
        f"Please note that you should first search for the target within the {color} rectangular bounding box. "
        "If you cannot find it, then search for the target outside the rectangular bounding box."
    )
