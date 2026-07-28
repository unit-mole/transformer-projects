from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"
SPACE_EVAL_DIR = PROJECT_ROOT / "space" / "evaluation"
CATEGORIES = ["color", "object", "counting", "yes_no", "action_scene", "spatial"]

COLORS = {
    "red": "#e63946",
    "blue": "#2455d6",
    "yellow": "#ffca28",
    "green": "#2fbf71",
    "orange": "#f28c28",
    "purple": "#8b5cf6",
    "pink": "#ec4899",
    "black": "#172033",
    "brown": "#9a6b3f",
    "gray": "#6b7280",
}


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, box: tuple[int, int, int, int], color: str) -> None:
    fill = COLORS[color]
    if shape == "circle":
        draw.ellipse(box, fill=fill, outline="#0b1526", width=4)
    elif shape == "square":
        draw.rounded_rectangle(box, radius=16, fill=fill, outline="#0b1526", width=4)
    elif shape == "triangle":
        x1, y1, x2, y2 = box
        points = [((x1 + x2) // 2, y1), (x2, y2), (x1, y2)]
        draw.polygon(points, fill=fill, outline="#0b1526")
    elif shape == "rectangle":
        draw.rounded_rectangle(box, radius=10, fill=fill, outline="#0b1526", width=4)
    elif shape == "star":
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        outer = min(x2 - x1, y2 - y1) / 2
        inner = outer * 0.42
        points = []
        import math
        for i in range(10):
            radius = outer if i % 2 == 0 else inner
            angle = -math.pi / 2 + i * math.pi / 5
            points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
        draw.polygon(points, fill=fill, outline="#0b1526")
    else:
        raise ValueError(f"Unsupported shape: {shape}")


def add_header(draw: ImageDraw.ImageDraw, title: str) -> None:
    draw.rounded_rectangle((18, 16, 622, 56), radius=12, fill="#edf4ff")
    draw.text((32, 29), title, fill="#233652")


def save_image(image: Image.Image, filename: str) -> None:
    for base in (ROOT_EVAL_DIR, SPACE_EVAL_DIR):
        image_dir = base / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        image.save(image_dir / filename, optimize=True)


def geometry_records() -> list[dict[str, object]]:
    scenes = [
        # left shape/color, right shape/color, extras, yes question/answer
        ("square", "blue", "circle", "red", 0, "Is the circle red?", ["yes"]),
        ("triangle", "yellow", "square", "green", 1, "Is the triangle blue?", ["no"]),
        ("circle", "orange", "triangle", "purple", 2, "Is the circle orange?", ["yes"]),
        ("rectangle", "pink", "star", "yellow", 3, "Is the star green?", ["no"]),
        ("star", "purple", "circle", "blue", 0, "Is the circle blue?", ["yes"]),
        ("square", "green", "triangle", "orange", 1, "Is the square red?", ["no"]),
        ("circle", "red", "rectangle", "gray", 2, "Is the rectangle gray?", ["yes"]),
        ("triangle", "blue", "star", "pink", 3, "Is the triangle yellow?", ["no"]),
        ("rectangle", "brown", "square", "yellow", 0, "Is the rectangle brown?", ["yes"]),
        ("star", "green", "circle", "purple", 1, "Is the circle orange?", ["no"]),
    ]
    records: list[dict[str, object]] = []
    number_words = {2: "two", 3: "three", 4: "four", 5: "five"}

    for index, (left_shape, left_color, right_shape, right_color, extras, yes_question, yes_answers) in enumerate(scenes, start=1):
        image = Image.new("RGB", (640, 420), "#f8fbff")
        draw = ImageDraw.Draw(image)
        add_header(draw, f"Synthetic geometry scene {index:02d}")
        draw.rectangle((0, 350, 640, 420), fill="#c8f7e4")
        draw_shape(draw, left_shape, (80, 125, 230, 275), left_color)
        draw_shape(draw, right_shape, (410, 125, 560, 275), right_color)

        extra_shapes = ["circle", "square", "triangle"]
        extra_colors = ["black", "orange", "pink"]
        for extra_index in range(extras):
            x = 255 + extra_index * 55
            draw_shape(draw, extra_shapes[extra_index], (x, 290, x + 38, 328), extra_colors[extra_index])

        filename = f"geometry_{index:02d}.png"
        save_image(image, filename)
        image_path = f"./evaluation/images/{filename}"
        total = 2 + extras

        records.extend(
            [
                {
                    "id": f"color-{index:02d}",
                    "image": image_path,
                    "question": f"What color is the {right_shape}?",
                    "accepted_answers": [right_color],
                    "category": "color",
                },
                {
                    "id": f"object-{index:02d}",
                    "image": image_path,
                    "question": "What shape is on the left?",
                    "accepted_answers": [left_shape],
                    "category": "object",
                },
                {
                    "id": f"counting-{index:02d}",
                    "image": image_path,
                    "question": "How many shapes are visible?",
                    "accepted_answers": [str(total), number_words[total]],
                    "category": "counting",
                },
                {
                    "id": f"yes-no-{index:02d}",
                    "image": image_path,
                    "question": yes_question,
                    "accepted_answers": yes_answers,
                    "category": "yes_no",
                },
                {
                    "id": f"spatial-{index:02d}",
                    "image": image_path,
                    "question": f"Which shape is to the right of the {left_shape}?",
                    "accepted_answers": [right_shape],
                    "category": "spatial",
                },
            ]
        )
    return records


def draw_person(draw: ImageDraw.ImageDraw, center_x: int, ground_y: int, pose: str = "standing") -> None:
    draw.ellipse((center_x - 22, ground_y - 155, center_x + 22, ground_y - 111), fill="#f2c094", outline="#0b1526", width=3)
    draw.line((center_x, ground_y - 110, center_x, ground_y - 45), fill="#0b1526", width=8)
    draw.line((center_x, ground_y - 80, center_x - 38, ground_y - 58), fill="#0b1526", width=7)
    draw.line((center_x, ground_y - 80, center_x + 38, ground_y - 58), fill="#0b1526", width=7)
    if pose == "kicking":
        draw.line((center_x, ground_y - 45, center_x - 28, ground_y), fill="#0b1526", width=8)
        draw.line((center_x, ground_y - 45, center_x + 50, ground_y - 20), fill="#0b1526", width=8)
    else:
        draw.line((center_x, ground_y - 45, center_x - 28, ground_y), fill="#0b1526", width=8)
        draw.line((center_x, ground_y - 45, center_x + 28, ground_y), fill="#0b1526", width=8)


def action_image(index: int, action: str) -> Image.Image:
    image = Image.new("RGB", (640, 420), "#dff4ff")
    draw = ImageDraw.Draw(image)
    add_header(draw, f"Synthetic action scene {index:02d}")
    draw.rectangle((0, 330, 640, 420), fill="#b9efc9")

    if action == "holding an umbrella":
        draw_person(draw, 280, 330)
        draw.arc((345, 90, 535, 245), 180, 360, fill="#7c3aed", width=16)
        draw.line((440, 170, 440, 310), fill="#0b1526", width=6)
    elif action == "reading a book":
        draw_person(draw, 300, 330)
        draw.polygon([(345, 205), (410, 220), (410, 285), (345, 270)], fill="#f59e0b", outline="#0b1526")
        draw.polygon([(410, 220), (475, 205), (475, 270), (410, 285)], fill="#fbbf24", outline="#0b1526")
    elif action == "riding a bicycle":
        draw.ellipse((120, 245, 250, 375), outline="#0b1526", width=10)
        draw.ellipse((390, 245, 520, 375), outline="#0b1526", width=10)
        draw.line((185, 310, 330, 225, 455, 310, 285, 310, 185, 310), fill="#2455d6", width=10)
        draw_person(draw, 330, 260)
    elif action == "chasing a ball":
        draw.ellipse((430, 270, 490, 330), fill="#e63946", outline="#0b1526", width=4)
        draw.ellipse((175, 245, 315, 325), fill="#9a6b3f", outline="#0b1526", width=4)
        draw.ellipse((265, 210, 340, 285), fill="#9a6b3f", outline="#0b1526", width=4)
        draw.line((195, 320, 170, 365), fill="#0b1526", width=7)
        draw.line((285, 320, 320, 365), fill="#0b1526", width=7)
    elif action == "flying":
        draw.arc((205, 150, 330, 265), 190, 350, fill="#172033", width=8)
        draw.arc((310, 150, 435, 265), 190, 350, fill="#172033", width=8)
        draw.ellipse((295, 205, 345, 250), fill="#6b7280", outline="#0b1526")
    elif action == "driving on a road":
        draw.rectangle((0, 250, 640, 420), fill="#6b7280")
        draw.line((0, 335, 640, 335), fill="#f8fafc", width=8)
        draw.rounded_rectangle((160, 220, 480, 335), radius=30, fill="#2455d6", outline="#0b1526", width=5)
        draw.polygon([(235, 220), (285, 160), (390, 160), (440, 220)], fill="#bde3ff", outline="#0b1526")
        draw.ellipse((210, 300, 285, 375), fill="#172033")
        draw.ellipse((365, 300, 440, 375), fill="#172033")
    elif action == "sailing on water":
        draw.rectangle((0, 250, 640, 420), fill="#5fc3e4")
        draw.polygon([(180, 290), (500, 290), (450, 360), (240, 360)], fill="#9a6b3f", outline="#0b1526")
        draw.line((340, 90, 340, 295), fill="#0b1526", width=8)
        draw.polygon([(345, 105), (345, 270), (500, 240)], fill="#f8fafc", outline="#0b1526")
    elif action == "kicking a ball":
        draw_person(draw, 290, 330, pose="kicking")
        draw.ellipse((430, 275, 495, 340), fill="#f8fafc", outline="#0b1526", width=4)
    elif action == "sitting on a chair":
        draw.rectangle((390, 200, 500, 310), fill="#9a6b3f", outline="#0b1526", width=5)
        draw.line((410, 310, 400, 380), fill="#0b1526", width=7)
        draw.line((480, 310, 490, 380), fill="#0b1526", width=7)
        draw.ellipse((250, 245, 410, 330), fill="#6b7280", outline="#0b1526", width=4)
        draw.ellipse((340, 205, 420, 285), fill="#6b7280", outline="#0b1526", width=4)
        draw.polygon([(350, 210), (365, 170), (385, 215)], fill="#6b7280", outline="#0b1526")
    elif action == "waving":
        draw_person(draw, 320, 330)
        draw.line((320, 250, 385, 155), fill="#0b1526", width=8)
        draw.ellipse((375, 135, 405, 165), fill="#f2c094", outline="#0b1526")
    else:
        raise ValueError(action)
    return image


def action_records() -> list[dict[str, object]]:
    actions = [
        ("holding an umbrella", "What is the person holding?", ["umbrella", "an umbrella"]),
        ("reading a book", "What is the person doing?", ["reading", "reading a book"]),
        ("riding a bicycle", "What is the person doing?", ["riding a bicycle", "cycling", "riding a bike"]),
        ("chasing a ball", "What is the dog doing?", ["chasing a ball", "chasing the ball"]),
        ("flying", "What is the bird doing?", ["flying"]),
        ("driving on a road", "What is the car doing?", ["driving", "driving on a road"]),
        ("sailing on water", "What is the boat doing?", ["sailing", "sailing on water"]),
        ("kicking a ball", "What is the person doing?", ["kicking a ball", "kicking the ball"]),
        ("sitting on a chair", "What is the cat doing?", ["sitting", "sitting on a chair"]),
        ("waving", "What is the person doing?", ["waving"]),
    ]
    records = []
    for index, (action, question, answers) in enumerate(actions, start=1):
        filename = f"action_{index:02d}.png"
        save_image(action_image(index, action), filename)
        records.append(
            {
                "id": f"action-scene-{index:02d}",
                "image": f"./evaluation/images/{filename}",
                "question": question,
                "accepted_answers": answers,
                "category": "action_scene",
            }
        )
    return records


def write_dataset(records: Iterable[dict[str, object]]) -> None:
    rows = list(records)
    for base in (ROOT_EVAL_DIR, SPACE_EVAL_DIR):
        base.mkdir(parents=True, exist_ok=True)
        with (base / "vqa_evaluation_60.json").open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
            handle.write("\n")
        with (base / "vqa_evaluation_60.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "image", "question", "accepted_answers", "category"])
            writer.writeheader()
            for row in rows:
                csv_row = dict(row)
                csv_row["accepted_answers"] = json.dumps(row["accepted_answers"])
                writer.writerow(csv_row)


def validate_dataset() -> None:
    source = ROOT_EVAL_DIR / "vqa_evaluation_60.json"
    if not source.exists():
        raise SystemExit(f"Missing {source}. Run this script without --check first.")
    records = json.loads(source.read_text(encoding="utf-8"))
    counts = Counter(str(row["category"]) for row in records)
    assert len(records) == 60, f"Expected 60 records, found {len(records)}"
    assert counts == Counter({category: 10 for category in CATEGORIES}), counts
    for row in records:
        relative = str(row["image"]).replace("./evaluation/", "")
        assert (ROOT_EVAL_DIR / relative).exists(), row["image"]
        assert (SPACE_EVAL_DIR / relative).exists(), row["image"]
    print("Evaluation dataset check passed: 60 records, 10 per category.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or validate the 60-pair synthetic VQA evaluation suite.")
    parser.add_argument("--check", action="store_true", help="Validate committed evaluation files without regenerating them.")
    args = parser.parse_args()
    if args.check:
        validate_dataset()
        return

    records = geometry_records() + action_records()
    write_dataset(records)
    validate_dataset()
    print(f"Generated evaluation assets under {ROOT_EVAL_DIR} and {SPACE_EVAL_DIR}.")


if __name__ == "__main__":
    main()
