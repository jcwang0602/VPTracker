PROMPT_VLT = """You are a visual object tracker.
Given the visual information of the target object, including an initial visual template <template><image></template>,
and the language description of the target object, including an initial language description <ref> <language> </ref>.

Then, detect the target object is in search image <search><image></search>.
    1. determine whether the target is visible in the search image.
    2. return the bounding box of its current position in the search image in the format [x_min, y_min, x_max, y_max].

Please return the answer in JSON format, for example: ```json\n{{\n  "visible": "yes/no", \n  "bbox":[x_min, y_min, x_max, y_max]\n}}\n```.
"""

PROMPT_VLT_VP = """You are a visual object tracker.
Given the visual information of the target object, including an initial visual template <template><image></template>,
and the language description of the target object, including an initial language description <ref> <language> </ref>.

Then, detect the target object is in search image <search><image></search>.
    1. determine whether the target is visible in the search image.
    2. return the bounding box of its current position in the search image in the format [x_min, y_min, x_max, y_max].

Please return the answer in JSON format, for example: ```json\n{{\n  "visible": "yes/no", \n  "bbox":[x_min, y_min, x_max, y_max]\n}}\n```.

Please note that you should first search for the target within the <vp_color> rectangular bounding box. If you cannot find it, then search for the target outside the rectangular bounding box."""

# Compact counterparts for training and inference experiments. Keep the image
# placeholders, language reference, output schema, and VP search priority.
PROMPT_VLT_COMPACT = (
    "Track <ref> <language> </ref> from <template><image></template> in "
    "<search><image></search>. Return JSON: "
    '{"visible":"yes/no","bbox":[x_min,y_min,x_max,y_max]}.'
)

PROMPT_VLT_BALANCED = (
    "You are a visual object tracker. Locate <ref> <language> </ref> using "
    "the template <template><image></template> in "
    "<search><image></search>. Decide visibility and return its current box "
    "as JSON: "
    '{"visible":"yes/no","bbox":[x_min,y_min,x_max,y_max]}.'
)

PROMPT_VLT_VP_COMPACT = (
    "Track <ref> <language> </ref> from <template><image></template> in "
    "<search><image></search>. Search inside the <vp_color> rectangle first, "
    "then outside. Return JSON: "
    '{"visible":"yes/no","bbox":[x_min,y_min,x_max,y_max]}.'
)

PROMPT_VLT_VP_BALANCED = (
    "You are a visual object tracker. Locate <ref> <language> </ref> using "
    "the template <template><image></template> in "
    "<search><image></search>. The <vp_color> rectangle is a candidate region "
    "from the previous target position, not the target box. Search inside it "
    "first; if the target is absent, search outside it. Decide visibility and "
    "return its current box as JSON: "
    '{"visible":"yes/no","bbox":[x_min,y_min,x_max,y_max]}.'
)


ANSWER_PROMPT_VISIBLE = (
    """```json\n{\n"visible": "<visible>", \n"bbox": <bbox>\n}\n```"""
)
ANSWER_PROMPT_INVISIBLE = """```json\n{\n"visible": "<visible>"}\n```"""
