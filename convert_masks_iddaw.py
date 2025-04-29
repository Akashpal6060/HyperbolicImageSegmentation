import os
import json
from PIL import Image, ImageDraw
from tqdm import tqdm

# ⚠️ Make sure this matches your iddaw_i2c.txt
LABEL2ID = {
    "road": 0,
    "parking": 1,
    "drivable_fallback": 2,
    "sidewalk": 3,
    "non_drivable_fallback": 4,
    "person": 5,
    "animal": 6,
    "rider": 7,
    "motorcycle": 8,
    "bicycle": 9,
    "auto_rickshaw": 10,
    "car": 11,
    "truck": 12,
    "bus": 13,
    "caravan": 14,
    "vehicle_fallback": 15,
    "curb": 16,
    "wall": 17,
    "fence": 18,
    "guard_rail": 19,
    "billboard": 20,
    "traffic_sign": 21,
    "traffic_light": 22,
    "pole": 23,
    "obs_str_bar_fallback": 24,
    "building": 25,
    "bridge": 26,
    "vegetation": 27,
    "sky": 28,
    "fallback_background": 29
}

def convert_single_json(json_path, png_path, width, height):
    with open(json_path, 'r') as f:
        data = json.load(f)

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    drawn = 0
    skipped = 0

    for obj in data.get("objects", []):
        label = obj.get("label")
        polygon = obj.get("polygon")
        class_id = LABEL2ID.get(label, None)

        if class_id is None or not polygon or len(polygon) < 3:
            skipped += 1
            continue

        polygon_points = [(int(x), int(y)) for x, y in polygon]
        draw.polygon(polygon_points, fill=class_id)
        drawn += 1

    mask.save(png_path)
    return drawn, skipped

def convert_all(json_root):
    conditions = ["FOG", "LOWLIGHT", "RAIN", "SNOW"]
    grand_total = 0

    for condition in conditions:
        print(f"\n🌦️ Processing condition: {condition}")
        cond_json_dir = os.path.join(json_root, "train", condition, "gtSeg")
        cond_png_dir = os.path.join(json_root, "train", condition, "gtSeg_png")
        os.makedirs(cond_png_dir, exist_ok=True)

        total_converted = 0

        for subfolder in sorted(os.listdir(cond_json_dir)):
            json_sub = os.path.join(cond_json_dir, subfolder)
            png_sub = os.path.join(cond_png_dir, subfolder)

            if not os.path.isdir(json_sub):
                continue  # ✅ skip non-directories like .stats.json.swn

            os.makedirs(png_sub, exist_ok=True)

            for fname in sorted(os.listdir(json_sub)):
                if not fname.endswith("_mask.json"):
                    continue

                json_path = os.path.join(json_sub, fname)
                png_path = os.path.join(png_sub, fname.replace("_mask.json", "_mask.png"))

                with open(json_path, "r") as f:
                    data = json.load(f)
                width = data["imgWidth"]
                height = data["imgHeight"]

                drawn, skipped = convert_single_json(json_path, png_path, width, height)
                print(f"  {fname} → ✅ mask.png | drawn: {drawn}, skipped: {skipped}")
                total_converted += 1

        print(f"📦 Total PNG masks for {condition}: {total_converted}")
        grand_total += total_converted

    print(f"\n🏁 Grand Total PNG masks across all conditions: {grand_total}")

if __name__ == "__main__":
    convert_all("datasets/IDDAW")
