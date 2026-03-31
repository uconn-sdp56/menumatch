import os
import json
import base64
from io import BytesIO
from typing import List

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from openai import OpenAI
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from .schemas import ClassificationResult, PortionEstimationResult


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

gpt_client = OpenAI(api_key=OPENAI_API_KEY)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_type = "DPT_Large"

MIDAS = torch.hub.load("intel-isl/MiDaS", model_type)
MIDAS.to(device)
MIDAS.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
MIDAS_TRANSFORM = transforms.dpt_transform

clipseg_model_id = "CIDAS/clipseg-rd64-refined"
CLIPSEG_PROCESSOR = CLIPSegProcessor.from_pretrained(clipseg_model_id)
CLIPSEG_MODEL = CLIPSegForImageSegmentation.from_pretrained(clipseg_model_id).to(device)
CLIPSEG_MODEL.eval()


def image_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def get_menu_items(hallid, meal, date):
    url = "https://husky-eats.onrender.com/api/menu"
    r = requests.get(url, params={"hallid": hallid, "meal": meal, "date": date})
    r.raise_for_status()
    return [
        {"name": item["name"], "id": item["id"]}
        for item in r.json()
        if "name" in item and "id" in item
    ]


def get_nutrition_info(item_id):
    url = "https://husky-eats.onrender.com/api/menuitem/" + str(item_id)
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


def gpt_item_classification(pil_img, menu_items, client, model="gpt-5-mini"):
    image_url = image_to_base64(pil_img)
    menu_items_payload = json.dumps(menu_items)

    prompt_text = (
        "You are a food identification expert specializing in dining hall meals.\n\n"
        "You will be shown:\n"
        "1) an image of a single plate\n"
        "2) the menu items available for that meal\n\n"
        "Your task: determine which provided menu items are actually present on the plate.\n\n"
        "PRIORITY:\n"
        "- Maximize precision on item presence.\n"
        "- Return the smallest set of menu items needed to explain the visible food.\n"
        "- Do not include weak, trace, garnish-like, or ambiguous items unless they clearly represent a meaningful serving.\n\n"
        "GUIDELINES:\n"
        "- Use the image as the primary evidence.\n"
        "- Only choose from the provided menu item list.\n"
        "- Do not invent new items.\n"
        "- Ignore trays, napkins, utensils, table surface, shadows, and reflections.\n"
        "- Prefer one strong match over multiple speculative matches.\n"
        "- If two candidate menu items are visually very similar, choose the one best supported by the image and menu context.\n"
        "- Do not include sauces, sides, or secondary items unless they are clearly visible as separate food on the plate.\n"
        "- If only a tiny residue or a few scattered grains are visible, usually exclude that item unless it clearly contributes a real portion.\n\n"
        "CONFIDENCE:\n"
        "- confidence should reflect how certain you are that the item is actually present on the plate.\n"
        "- Use lower confidence for ambiguous items, partial visibility, or visually similar alternatives.\n\n"
        "OUTLIERS:\n"
        "- If you have to decide between Roasted Chicken and Grilled Chicken Breast, Roasted Chicken is usually chopped, Grilled Chicken Breasts usually come in pieces"
        "OUTPUT FORMAT (strict JSON):\n"
        '- "items": an array of objects, each with fields {"id", "name", "confidence"}\n'
        '- "confidence": a float in [0,1] for each chosen item\n'
        '- "explanation": a brief 1-2 sentence rationale describing the key visual evidence\n'
    )

    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {"type": "input_text", "text": menu_items_payload},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
        text_format=ClassificationResult,
    )

    return response.output_parsed.model_dump()


def overlay_mask(img, mask, alpha=0.35):
    color = np.zeros_like(img)
    color[..., 1] = 255
    return np.where(mask[..., None] > 0, (img * (1 - alpha) + color * alpha).astype(np.uint8), img)


def _clipseg_prompt_mask(pil_img, prompt, threshold=0.5):
    inputs = CLIPSEG_PROCESSOR(text=[prompt], images=[pil_img], padding=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = CLIPSEG_MODEL(**inputs)

    logits = outputs.logits[0]
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    prob_image = Image.fromarray((probs * 255).astype(np.uint8), mode="L")
    prob_image = prob_image.resize(pil_img.size, Image.Resampling.BILINEAR)
    prob_mask = np.array(prob_image).astype(np.float32) / 255.0
    binary_mask = (prob_mask >= threshold).astype(np.uint8)
    return prob_mask, binary_mask


def _keep_largest_component(mask_u8):
    mask_u8 = (mask_u8 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    if num_labels <= 1:
        return mask_u8

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8)


def _fill_mask_holes(mask_u8):
    mask_u8 = (mask_u8 > 0).astype(np.uint8)
    h, w = mask_u8.shape

    flood = (mask_u8 * 255).copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask_u8 * 255, holes)
    return (filled > 0).astype(np.uint8)


def _clean_plate_mask(mask_u8, kernel_size=11):
    mask_u8 = (mask_u8 > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    largest = _keep_largest_component(opened)
    filled = _fill_mask_holes(largest)
    return filled.astype(np.uint8)


def detect_plate_mask(image_rgb: np.ndarray, prompt="plate", threshold=0.5, kernel_size=11) -> np.ndarray:
    image_rgb = np.asarray(image_rgb)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image_rgb.shape}")

    pil_img = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
    _, binary_mask = _clipseg_prompt_mask(pil_img, prompt=prompt, threshold=threshold)
    cleaned_mask = _clean_plate_mask(binary_mask, kernel_size=kernel_size)

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return (cleaned_mask * 255).astype(np.uint8)

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return (cleaned_mask * 255).astype(np.uint8)

    ellipse = cv2.fitEllipse(contour)
    ellipse_mask = np.zeros_like(cleaned_mask, dtype=np.uint8)
    cv2.ellipse(ellipse_mask, ellipse, 255, thickness=-1)
    return ellipse_mask


def _to_u8_mask(m):
    m = (m > 0).astype(np.uint8)
    return m


def ellipse_params_from_mask(plate_mask_u8):
    cnts, _ = cv2.findContours(plate_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise RuntimeError("No contour found in plate_mask.")
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        raise RuntimeError("Not enough contour points to fit ellipse.")
    (cx, cy), (w, h), angle = cv2.fitEllipse(c)
    plate_diameter_px = float(max(w, h))
    return (float(cx), float(cy), plate_diameter_px, ((cx, cy), (w, h), angle))


def empty_plate_height_map(
    *,
    height_px: int,
    width_px: int,
    center_xy_px: tuple[float, float],
    plate_diameter_px: float,
    outer_diameter_mm: float,
    inner_radius_mm: float,
    lip_end_radius_mm: float,
    slope_end_radius_mm: float,
    h_inner_mm: float = 0.0,
    h_lip_mm: float | None = None,
    h_slope_end_mm: float = 0.0,
    lip_profile: str = "smooth",
    slope_profile: str = "smooth",
    feather_px: float = 2.0,
    eps: float = 1e-6,
):
    if h_lip_mm is None:
        h_lip_mm = h_inner_mm

    mm_per_px = outer_diameter_mm / plate_diameter_px
    outer_radius_mm = outer_diameter_mm / 2.0

    if not (0 <= inner_radius_mm <= lip_end_radius_mm <= slope_end_radius_mm <= outer_radius_mm):
        raise ValueError("Radii must satisfy: 0 <= inner <= lip_end <= slope_end <= outer.")

    cx, cy = center_xy_px

    ys, xs = np.indices((height_px, width_px), dtype=np.float32)
    dx_px = xs - cx
    dy_px = ys - cy
    r_mm = np.sqrt((dx_px * mm_per_px) ** 2 + (dy_px * mm_per_px) ** 2)

    def smoothstep01(t):
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def ramp(t, mode):
        if mode == "smooth":
            return smoothstep01(t)
        if mode == "linear":
            return np.clip(t, 0.0, 1.0)
        if mode == "flat":
            return (t >= 0).astype(np.float32)
        raise ValueError("profile must be 'smooth', 'linear', or 'flat'")

    def soft_inside(r, R):
        if feather_px <= 0:
            return (r <= R).astype(np.float32)
        feather_mm = feather_px * mm_per_px
        lo = R - feather_mm / 2.0
        hi = R + feather_mm / 2.0
        t = (hi - r) / (hi - lo)
        return np.clip(t, 0.0, 1.0).astype(np.float32)

    inside_inner = soft_inside(r_mm, inner_radius_mm)
    inside_lip_end = soft_inside(r_mm, lip_end_radius_mm)
    inside_slope_end = soft_inside(r_mm, slope_end_radius_mm)
    inside_outer = soft_inside(r_mm, outer_radius_mm)

    lip_ring = np.clip(inside_lip_end - inside_inner, 0.0, 1.0)
    slope_ring = np.clip(inside_slope_end - inside_lip_end, 0.0, 1.0)
    outer_band = np.clip(inside_outer - inside_slope_end, 0.0, 1.0)

    Z_mm = np.zeros((height_px, width_px), dtype=np.float32)

    Z_mm += h_inner_mm * inside_inner

    if lip_end_radius_mm > inner_radius_mm:
        t = (r_mm - inner_radius_mm) / (lip_end_radius_mm - inner_radius_mm)
        s = ramp(t, lip_profile)
        Z_lip = h_inner_mm + (h_lip_mm - h_inner_mm) * s
        Z_mm += Z_lip * lip_ring

    if slope_end_radius_mm > lip_end_radius_mm:
        t = (r_mm - lip_end_radius_mm) / (slope_end_radius_mm - lip_end_radius_mm)
        s = ramp(t, slope_profile)
        Z_slope = h_lip_mm + (h_slope_end_mm - h_lip_mm) * s
        Z_mm += Z_slope * slope_ring

    Z_mm += h_slope_end_mm * outer_band

    Z_mm *= inside_outer

    zmax = float(Z_mm.max())
    if zmax <= eps:
        return np.zeros_like(Z_mm, dtype=np.float32)

    Z_rel = (Z_mm / zmax).astype(np.float32)

    Z_rel *= inside_outer

    return Z_mm, Z_rel


def run_midas(midas, midas_transform, img):
    input_batch = midas_transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

    depth = prediction.squeeze().cpu().numpy()
    depth = cv2.resize(depth, (img.shape[1], img.shape[0]))
    return depth


def _validate_calibration_inputs(midas_depth, food_union_mask, plate_mask, empty_plate_mm_map):
    depth = np.asarray(midas_depth, dtype=np.float32)
    food = (np.asarray(food_union_mask) > 0).astype(np.uint8)
    plate = (np.asarray(plate_mask) > 0).astype(np.uint8)
    plate_mm = np.asarray(empty_plate_mm_map, dtype=np.float32)

    if depth.shape != food.shape or depth.shape != plate.shape or depth.shape != plate_mm.shape:
        raise ValueError(
            f"All calibration inputs must have the same shape, got "
            f"{depth.shape=}, {food.shape=}, {plate.shape=}, {plate_mm.shape=}"
        )

    if plate.sum() == 0:
        raise RuntimeError("Plate mask is empty; cannot calibrate MiDaS depth.")

    return depth, food, plate, plate_mm


def _build_table_ring_mask(plate_mask, ring_px=25):
    plate = (np.asarray(plate_mask) > 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_px, ring_px))
    plate_dilated = cv2.dilate(plate, kernel)
    table_ring = (plate_dilated > 0) & (plate == 0)

    if table_ring.sum() == 0:
        table_ring = plate == 0

    if table_ring.sum() == 0:
        raise RuntimeError("No table/background pixels available for MiDaS calibration.")

    return table_ring.astype(np.uint8)


def _zero_depth_to_table(midas_depth, table_ring_mask):
    depth = np.asarray(midas_depth, dtype=np.float32)
    table_ring = np.asarray(table_ring_mask) > 0

    if table_ring.sum() == 0:
        raise RuntimeError("Table ring mask is empty; cannot estimate table depth level.")

    table_level = float(np.median(depth[table_ring]))
    depth_zeroed = depth - table_level
    return depth_zeroed, table_level


def _build_visible_plate_mask(plate_mask, food_union_mask, erode_px=5):
    plate = (np.asarray(plate_mask) > 0).astype(np.uint8)
    food = (np.asarray(food_union_mask) > 0).astype(np.uint8)

    visible_plate = ((plate > 0) & (food == 0)).astype(np.uint8)
    if erode_px > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
        visible_plate = cv2.erode(visible_plate, kernel)

    if visible_plate.sum() == 0:
        raise RuntimeError("No visible plate pixels remain after removing food.")

    return visible_plate


def _estimate_midas_units_per_mm(depth_zeroed, visible_plate_mask, empty_plate_mm_map, min_plate_mm=3.0):
    depth_zeroed = np.asarray(depth_zeroed, dtype=np.float32)
    visible_plate = np.asarray(visible_plate_mask) > 0
    plate_mm = np.asarray(empty_plate_mm_map, dtype=np.float32)

    scale_mask = visible_plate & (plate_mm >= min_plate_mm)
    if scale_mask.sum() < 50:
        scale_mask = visible_plate & (plate_mm > 1.0)

    if scale_mask.sum() == 0:
        raise RuntimeError("Not enough visible plate pixels with non-trivial plate height to estimate metric scale.")

    ratios = depth_zeroed[scale_mask] / np.clip(plate_mm[scale_mask], 1e-6, None)
    ratios = ratios[np.isfinite(ratios)]
    ratios = ratios[ratios > 0]

    if ratios.size == 0:
        raise RuntimeError("Failed to estimate a positive MiDaS-units-per-mm scale from the visible plate.")

    midas_units_per_mm = float(np.median(ratios))
    return midas_units_per_mm, scale_mask.astype(np.uint8), ratios


def _convert_zeroed_depth_to_mm(depth_zeroed, midas_units_per_mm):
    depth_zeroed = np.asarray(depth_zeroed, dtype=np.float32)

    if midas_units_per_mm <= 0:
        raise RuntimeError("midas_units_per_mm must be positive.")

    return depth_zeroed / midas_units_per_mm


def _subtract_empty_plate_geometry(depth_mm, empty_plate_mm_map):
    depth_mm = np.asarray(depth_mm, dtype=np.float32)
    plate_mm = np.asarray(empty_plate_mm_map, dtype=np.float32)
    return depth_mm - plate_mm


def _finalize_food_height_map(residual_mm, visible_plate_mask, plate_mask, noise_percentile=80):
    residual_mm = np.asarray(residual_mm, dtype=np.float32)
    visible_plate = np.asarray(visible_plate_mask) > 0
    plate = np.asarray(plate_mask) > 0

    if visible_plate.sum() > 0:
        noise_floor_mm = float(np.percentile(np.maximum(residual_mm[visible_plate], 0.0), noise_percentile))
    else:
        noise_floor_mm = 0.0

    calibrated_mm = np.maximum(residual_mm - noise_floor_mm, 0.0)
    calibrated_mm *= plate.astype(np.float32)
    return calibrated_mm, noise_floor_mm


def build_calibrated_food_height_map(midas_depth, food_union_mask, plate_mask, empty_plate_mm_map):
    depth, food, plate, plate_mm = _validate_calibration_inputs(
        midas_depth=midas_depth,
        food_union_mask=food_union_mask,
        plate_mask=plate_mask,
        empty_plate_mm_map=empty_plate_mm_map,
    )

    table_ring_mask = _build_table_ring_mask(plate_mask=plate)
    depth_zeroed, _ = _zero_depth_to_table(midas_depth=depth, table_ring_mask=table_ring_mask)
    visible_plate_mask = _build_visible_plate_mask(plate_mask=plate, food_union_mask=food)
    midas_units_per_mm, _, _ = _estimate_midas_units_per_mm(
        depth_zeroed=depth_zeroed,
        visible_plate_mask=visible_plate_mask,
        empty_plate_mm_map=plate_mm,
    )
    depth_mm = _convert_zeroed_depth_to_mm(depth_zeroed=depth_zeroed, midas_units_per_mm=midas_units_per_mm)
    residual_mm = _subtract_empty_plate_geometry(depth_mm=depth_mm, empty_plate_mm_map=plate_mm)
    calibrated_mm, _ = _finalize_food_height_map(
        residual_mm=residual_mm,
        visible_plate_mask=visible_plate_mask,
        plate_mask=plate,
    )
    return calibrated_mm


def segment_prompt(image, prompt, threshold=0.5):
    inputs = CLIPSEG_PROCESSOR(text=[prompt], images=[image], padding=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = CLIPSEG_MODEL(**inputs)

    logits = outputs.logits[0]
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    prob_image = Image.fromarray((probs * 255).astype(np.uint8), mode="L")
    prob_image = prob_image.resize(image.size, Image.Resampling.BILINEAR)
    prob_mask = np.array(prob_image).astype(np.float32) / 255.0
    binary_mask = (prob_mask >= threshold).astype(np.uint8)
    return prob_mask, binary_mask


def calculate_food_volume(depth_mm, plate_mask, food_mask, plate_diameter_mm):
    depth = np.asarray(depth_mm, dtype=np.float32)
    plate = np.asarray(plate_mask, dtype=np.float32)
    food = np.asarray(food_mask, dtype=np.float32)

    if depth.shape != plate.shape or depth.shape != food.shape:
        raise ValueError(
            f"depth_mm, plate_mask, and food_mask must have the same shape, got {depth.shape}, {plate.shape}, {food.shape}"
        )

    if plate_diameter_mm <= 0:
        raise ValueError("plate_diameter_mm must be > 0")

    depth = np.maximum(depth, 0.0)
    plate = np.clip(plate, 0.0, 1.0)
    food = np.clip(food, 0.0, 1.0)

    food = food * (plate > 0)

    plate_area_mm2 = np.pi * (plate_diameter_mm * 0.5) ** 2
    plate_area_px = float(plate.sum())

    if plate_area_px <= 1e-6:
        return 0.0, 0.0, 0.0

    pixel_area_mm2 = plate_area_mm2 / plate_area_px
    volume_mm3 = float((depth * food).sum()) * float(pixel_area_mm2)
    volume_ml = volume_mm3 / 1000.0

    return volume_ml, volume_mm3, pixel_area_mm2


OUTER_DIAMETER_MM = 272.5
INNER_RADIUS_MM = 177 / 2
LIP_END_RADIUS_MM = 188 / 2
SLOPE_END_RADIUS_MM = 272.5 / 2
H_INNER_MM = 9.0
H_LIP_MM = 13.0
H_SLOPE_END_MM = 20.0
LIP_PROFILE = "smooth"
SLOPE_PROFILE = "linear"
FEATHER_PX = 2.0


def estimate_item_volumes(pil_img, classification_result):
    img = np.array(pil_img)
    plate_mask_u8 = _to_u8_mask(detect_plate_mask(img))

    height_px, width_px = plate_mask_u8.shape
    cx, cy, plate_diameter_px, _ = ellipse_params_from_mask(plate_mask_u8)

    empty_plate_mm_map, _ = empty_plate_height_map(
        height_px=height_px,
        width_px=width_px,
        center_xy_px=(cx, cy),
        plate_diameter_px=plate_diameter_px,
        outer_diameter_mm=OUTER_DIAMETER_MM,
        inner_radius_mm=INNER_RADIUS_MM,
        lip_end_radius_mm=LIP_END_RADIUS_MM,
        slope_end_radius_mm=SLOPE_END_RADIUS_MM,
        h_inner_mm=H_INNER_MM,
        h_lip_mm=H_LIP_MM,
        h_slope_end_mm=H_SLOPE_END_MM,
        lip_profile=LIP_PROFILE,
        slope_profile=SLOPE_PROFILE,
        feather_px=FEATHER_PX,
    )

    midas_depth = run_midas(MIDAS, MIDAS_TRANSFORM, img)

    food_masks = {}
    for item in classification_result["items"]:
        _, binary_mask = segment_prompt(image=pil_img, prompt=item["name"], threshold=0.5)
        food_masks[item["id"]] = binary_mask.astype(np.uint8)

    food_union_mask = np.zeros((height_px, width_px), dtype=np.uint8)
    for binary_mask in food_masks.values():
        food_union_mask = np.maximum(food_union_mask, binary_mask)

    calibrated_food_height_mm = build_calibrated_food_height_map(
        midas_depth=midas_depth,
        food_union_mask=food_union_mask,
        plate_mask=plate_mask_u8,
        empty_plate_mm_map=empty_plate_mm_map,
    )

    result = {}
    for item in classification_result["items"]:
        volume_ml, _, _ = calculate_food_volume(
            calibrated_food_height_mm,
            plate_mask_u8,
            food_masks[item["id"]],
            OUTER_DIAMETER_MM,
        )
        result[item["id"]] = volume_ml

    return result


def gpt_portion_estimation(
    pil_img,
    classification_result,
    classification_volumes_by_item,
    client,
    model="gpt-5-mini",
):
    image_url = image_to_base64(pil_img)

    payload_obj = {
        "items": classification_result["items"],
        "classification_volumes_ml": (
            None
            if classification_volumes_by_item is None
            else {
                str(item_id): float(volume_ml)
                for item_id, volume_ml in classification_volumes_by_item.items()
            }
        ),
    }
    payload = json.dumps(payload_obj, indent=2)

    prompt_text = (
        "You are a nutrition analyst estimating portion sizes from a single plate image.\n\n"
        "INPUTS:\n"
        "1) An image of a single plate.\n"
        "2) A JSON of detected items with nutrition info, including serving size.\n"
        "3) classification_volumes_ml: either an estimated volume in mL for each detected item, keyed by item id, or null if volume estimation failed.\n\n"
        "TASK:\n"
        "- For each detected item, estimate the number of servings on the plate.\n"
        "- Only return items present in the provided JSON.\n"
        "- Use conservative judgment when evidence is weak or conflicting.\n\n"
        "CRITICAL RULE: VOLUME IS A WEAK SIGNAL, NOT GROUND TRUTH.\n"
        "- Never trust classification_volumes_ml blindly.\n"
        "- Ignore a volume estimate if it is zero or near zero for clearly visible food.\n"
        "- Ignore a volume estimate if it is implausibly large or implausibly small for what is visible.\n"
        "- Ignore a volume estimate if it strongly conflicts with obvious visual evidence such as piece count, footprint, or relative portion size.\n"
        "- If volume looks noisy but directionally useful, treat it as a weak prior, not an absolute rule.\n\n"
        "WHEN TO PREFER VISUAL ESTIMATION OVER VOLUME:\n"
        "- Prefer visual estimation for discrete countable foods such as nuggets, tenders, wings, breadsticks, and similar piece-based foods.\n"
        "- Prefer visual estimation for sauces, shredded meats, mixed stir-fries, leafy vegetables, and irregular foods whose apparent footprint can be misleading.\n"
        "- For EACH items, count pieces first and use volume only as a weak sanity check.\n\n"
        "WHEN VOLUME CAN HELP:\n"
        "- Volume can be useful for amorphous foods such as rice, beans, mashed potatoes, macaroni and cheese, and corn.\n"
        "- Even for those foods, ignore volume if it looks inconsistent with the image.\n\n"
        "SERVING SIZE RULES:\n"
        "- If serving size uses EACH and shows a number N (example: '4 EACH'), estimate visible piece count P and report servings = P / N.\n"
        "- For EACH items, piece count is the primary signal.\n"
        "- For non-EACH items with gram or ounce serving sizes, you may convert plausible volume to estimated weight using a reasonable density heuristic, then divide by the serving-size weight.\n"
        "- For cup-based serving sizes, you may reason from visible volume directly, but still sanity-check against the image.\n\n"
        "DENSITY HEURISTICS (ONLY WHEN VOLUME IS PLAUSIBLE):\n"
        "- watery foods / sauces / soups / beans: about 0.9-1.1 g/mL\n"
        "- cooked rice / pasta / mac and cheese / mashed foods: about 0.6-0.9 g/mL\n"
        "- fries / nuggets / roasted vegetables: about 0.35-0.7 g/mL\n"
        "- solid meats / dense mixed dishes: about 0.8-1.1 g/mL\n\n"
        "ANTI-DOUBLE-COUNTING RULE:\n"
        "- Do not assign the same visible food mass to multiple items.\n"
        "- If two candidate items seem to refer to the same region, assign servings only to the better-supported item.\n"
        "- Be especially cautious with side items, sauces, and small accompaniments.\n\n"
        "CONSERVATIVE OUTPUT RULES:\n"
        "- Avoid large serving estimates unless the image clearly supports them.\n"
        "- If evidence is mixed, choose the more conservative plausible estimate.\n"
        "- If a volume estimate is clearly bad, say in the explanation that you ignored it.\n\n"
        "OUTPUT RULES:\n"
        "- Do not invent items.\n"
        "- Do not output piece count as a field.\n"
        "- Include brief reasoning in explanation; for EACH items include counted pieces there when relevant.\n\n"
        "FORMAT (strict JSON):\n"
        "{\n"
        '  "servings": [ { "id": <int>, "name": "<str>", "num_servings": <float> }, ... ],\n'
        '  "explanation": "<1-2 short sentences>"\n'
        "}\n"
    )

    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {"type": "input_text", "text": payload},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
        text_format=PortionEstimationResult,
    )

    return response.output_parsed.model_dump()


def predict(pil_image, dining_hall_id, meal, date):
    menu_items = get_menu_items(dining_hall_id, meal, date)
    classification_result = gpt_item_classification(pil_image, menu_items, gpt_client)

    try:
        item_volumes = estimate_item_volumes(pil_image, classification_result)
    except Exception:
        item_volumes = None

    for item in classification_result["items"]:
        item["nutrition"] = get_nutrition_info(item["id"])

    portion_result = gpt_portion_estimation(pil_image, classification_result, item_volumes, gpt_client)
    return portion_result["servings"]