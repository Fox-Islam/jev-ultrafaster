"""The viewport as an image for TypeSafe, with the offered controls outlined and numbered.

Off unless JEV_IMAGES is set: images are a preview feature of the API, sent only with its opt-in
header, and a 1120x780 screenshot costs about 2.5k input tokens per decision. JEV_IMAGES=1 sends one
with every decision; JEV_IMAGES=auto only where `wanted` or `second_look` is true.
"""

import base64
import io
import os
import re
import uuid

HEADERS = {"TypeSafe-Danger-Zone-Images": "true"}

# What the API accepts, inclusive. An image outside these is rejected, not resized, and one rejected
# image fails the whole request, so it is fitted here or not sent at all.
SIDE = (256, 1568)
PIXELS = (78_400, 2_458_624)

INK, PAPER = (255, 0, 170), (255, 255, 255)


def mode():
    return {"1": "always", "auto": "auto"}.get(os.environ.get("JEV_IMAGES", ""), "off")


def enabled():
    return mode() != "off"


# Goals that name what something looks like or where it sits. On fixtures where the colour or the
# highlight a goal named was carried only by styling, text-only chose the right control in 3 of 8
# runs, and with the screenshot in 8 of 8.
VISUAL = re.compile(
    r"\b(colou?r\w*|red|green|blue|yellow|orange|purple|pink|black|white|gr[ae]y|highlight\w*|icons?|images?|"
    r"pictures?|photos?|logos?|layout|overlap\w*|broken|styl(e|ed|ing)|(left|right) of|next to)\b",
    re.IGNORECASE,
)

# Where a text-only answer is asked again with the screenshot. On two real tasks every operation
# but DONE came at a confidence of 0.86-0.99; on the icon and tab fixtures, where the answer was
# only visible, at 0.35-0.57 whether right or wrong. A wrong swatch came at 0.75-0.84, which only
# `unnamed_row` catches. A false DONE on a page whose failure showed only in the screenshot came at
# 0.60-0.68 and right DONEs at 0.65-0.95, so below 0.9 confidence cannot separate them.
UNCERTAIN = 0.6
DONE_SURE = 0.9


def unnamed_row(actions):
    """Whether two or more unnamed controls in view sit together, as an icon toolbar or a set of
    swatches do. Text-only chose the right one from such a row in 1 of 8 runs. An unnamed control
    alone does not count: of eleven real pages, one had four alone and none had two together."""
    seen, groups = set(), {}
    for action in actions:
        if action.get("group") is None or action.get("offscreen") or action["node"] in seen:
            continue
        seen.add(action["node"])
        groups[action["group"]] = groups.get(action["group"], 0) + 1
    return any(count >= 2 for count in groups.values())


def wanted(look, actions, goal):
    """Whether this decision is asked with the screenshot from the start. `look` is True for a
    decision being asked again; see `second_look`."""
    if mode() == "always":
        return True
    return mode() == "auto" and bool(look or unnamed_row(actions) or VISUAL.search(goal or ""))


def second_look(operation, confidence):
    """Whether a text-only answer is asked again with the screenshot: it ends the run, or the
    model was unsure which operation to take."""
    confidence = confidence or 0
    return (
        operation == "BLOCKED"
        or operation == "DONE" and confidence < DONE_SURE
        or operation != "DONE" and confidence < UNCERTAIN
    )


def require():
    """Fail at start instead of at the first decision, when a run has already begun."""
    if not enabled():
        return
    try:
        import PIL  # noqa: F401
    except ImportError:
        raise RuntimeError("JEV_IMAGES needs Pillow: uv sync --extra images") from None


def placed(action):
    """Where an action's element is, in page coordinates, or None if it has no place on screen."""
    rect = action.get("rect")
    if not rect or action.get("offscreen"):
        return None
    dx, dy = action.get("offset") or (0.0, 0.0)
    return rect["x"] + dx, rect["y"] + dy, rect["w"], rect["h"]


def places(targets):
    """One rectangle per element index, from whichever of its actions has one."""
    found = {}
    for candidates in targets.values():
        for key, action in candidates.items():
            index = key.split(":")[0]
            if index not in found and (where := placed(action)):
                found[index] = where
    return found


def marked(page, targets):
    """The page's screenshot as JPEG bytes, each element in view outlined and tagged with its
    index, or None when there is no screenshot or it cannot be fitted to the API's limits.

    Drawn on the captured image, never on the page: marking the page would change what the
    decision is checked against and what the next reading sees.
    """
    if not page.get("screenshot") or not page.get("w"):
        return None
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(io.BytesIO(base64.b64decode(page["screenshot"]))).convert("RGB")
    longest = max(image.size)
    if longest > SIDE[1]:
        image = image.resize((round(image.width * SIDE[1] / longest), round(image.height * SIDE[1] / longest)))
    if min(image.size) < SIDE[0] or not PIXELS[0] <= image.width * image.height <= PIXELS[1]:
        return None
    scale = image.width / page["w"]
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(11, round(13 * scale)))
    for index, (x, y, w, h) in places(targets).items():
        box = [x * scale, y * scale, (x + w) * scale, (y + h) * scale]
        if box[2] <= 0 or box[3] <= 0 or box[0] >= image.width or box[1] >= image.height:
            continue
        draw.rectangle(box, outline=INK, width=2)
        left, top, right, bottom = draw.textbbox((0, 0), index, font=font)
        tag = [max(0, box[0]), max(0, box[1]), max(0, box[0]) + right - left + 4, max(0, box[1]) + bottom - top + 2]
        draw.rectangle(tag, fill=INK)
        draw.text((tag[0] + 2 - left, tag[1] + 1 - top), index, fill=PAPER, font=font)
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=80, progressive=False)
    return out.getvalue()


def attach(body, image):
    """Put the image in the request's state and assets. Returns the placeholder it is named by.

    A UUID, because placeholders are matched as plain text anywhere in the request, and a short
    name would also match inside page text.
    """
    name = f"screen-{uuid.uuid4()}"
    body["state"]["screen"] = name
    body["assets"] = {name: {"type": "image", "format": "jpg", "data_base64": base64.b64encode(image).decode()}}
    return name


def recorded(body):
    """The request as kept in a decision: the image's size in place of the image."""
    if "assets" not in body:
        return body
    return {
        **body,
        "assets": {
            name: {"type": "image", "format": asset["format"], "bytes": len(base64.b64decode(asset["data_base64"]))}
            for name, asset in body["assets"].items()
        },
    }
