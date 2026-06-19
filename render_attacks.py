import argparse
import json
import os

from PIL import Image
from pytorch_lightning import seed_everything

from utils.typo_attack_planner import TypoAttackPlanner


def default_render_dir(base_dir, dataset, slider, filter_value, seed):
    render_dir = os.path.join(base_dir, dataset, f"slider_{slider}", f"filter_{filter_value}", f"seed_{seed}")
    os.makedirs(render_dir, exist_ok=True)
    return render_dir


def seg_image_name(image_name):
    stem, ext = os.path.splitext(image_name)
    return f"{stem}_seg{ext}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, default=None)
    parser.add_argument("--render-dir", type=str, default=None)
    parser.add_argument("--render-base-dir", type=str, default="./attack_renders")
    parser.add_argument("--scale-factor", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)

    with open(args.plan_file, "r", encoding="utf-8") as f:
        plan_payload = json.load(f)

    meta = plan_payload["meta"]
    records = plan_payload["records"]
    image_folder = args.image_folder or meta["image_folder"]
    som_image_folder = meta["som_image_folder"]

    render_dir = args.render_dir
    if render_dir is None:
        render_dir = default_render_dir(
            base_dir=args.render_base_dir,
            dataset=meta["dataset"],
            slider=meta["slider"],
            filter_value=meta["filter"],
            seed=meta["seed"],
        )

    image_save_dir = os.path.join(render_dir, "images")
    diffusion_dir = os.path.join(render_dir, "diffusion")
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(diffusion_dir, exist_ok=True)
    output_file = os.path.join(render_dir, "renders.json")

    existing_records = []
    completed_question_ids = set()
    if args.resume and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            existing_payload = json.load(f)
        existing_records = existing_payload.get("records", [])
        completed_question_ids = {record["question_id"] for record in existing_records}

    planner = TypoAttackPlanner(som_image_folder=som_image_folder, load_diffuser=True)

    rendered_records = list(existing_records)
    for record in records:
        question_id = record["question_id"]
        if question_id in completed_question_ids:
            continue

        image_name = record["image_original"]
        image_path = os.path.join(image_folder, image_name)
        image_save_name = record["image_save_name"]

        diffusion_images = planner.render_from_plan(
            image_path=image_path,
            plan_detail=record["plan_detail"],
            rectangle=record["rectangle"],
            scale_factor=args.scale_factor,
        )
        attack_image = diffusion_images[0]
        attack_image_path = os.path.join(image_save_dir, image_save_name)
        attack_image.save(attack_image_path)

        _, seg_image, _ = planner._load_som_assets(image_path)
        seg_image_path = os.path.join(image_save_dir, seg_image_name(image_save_name))
        seg_image.save(seg_image_path)

        question_diffusion_dir = os.path.join(diffusion_dir, str(question_id))
        os.makedirs(question_diffusion_dir, exist_ok=True)
        diffusion_image_paths = []
        for idx, image in enumerate(diffusion_images):
            diffusion_image_path = os.path.join(question_diffusion_dir, f"{idx}.jpg")
            image.save(diffusion_image_path)
            diffusion_image_paths.append(diffusion_image_path)

        rendered_record = dict(record)
        rendered_record.update({
            "attack_image_path": attack_image_path,
            "seg_image_path": seg_image_path,
            "diffusion_image_paths": diffusion_image_paths,
        })
        rendered_records.append(rendered_record)

        payload = {
            "meta": {
                **meta,
                "render_dir": render_dir,
                "image_folder": image_folder,
            },
            "records": rendered_records,
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"[render] saved question_id={question_id}")
