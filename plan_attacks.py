import argparse
import json
import os

from pytorch_lightning import seed_everything

from utils.typo_attack_planner import TypoAttackPlanner


def build_som_image_folder(base_path, dataset, slider, seed, filter_value):
    return os.path.join(base_path, dataset, f"slider_{slider}", f"seed_{seed}", f"filter_{filter_value}")


def build_image_save_name(dataset, image_name, question_id):
    if "vqav2" in dataset or "LingoQA" in dataset:
        stem, ext = os.path.splitext(image_name)
        return f"{stem}_{question_id}{ext}"
    return image_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="typo_base_complex")
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--som-base-path", type=str, default="./som_images")
    parser.add_argument("--som-image-folder", type=str, default=None)
    parser.add_argument("--output-file", type=str, default="./attack_plans/plans.json")
    parser.add_argument("--slider", type=float, default=3)
    parser.add_argument("--filter", type=float, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planner-model", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--planner-base-url", type=str, default=None)
    parser.add_argument("--planner-api-key", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=4095)
    parser.add_argument("--top_p", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)

    som_image_folder = args.som_image_folder
    if som_image_folder is None:
        som_image_folder = build_som_image_folder(
            base_path=args.som_base_path,
            dataset=args.dataset,
            slider=args.slider,
            seed=args.seed,
            filter_value=args.filter,
        )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.question_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    existing_records = []
    completed_question_ids = set()
    if args.resume and os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            existing_payload = json.load(f)
        existing_records = existing_payload.get("records", [])
        completed_question_ids = {record["question_id"] for record in existing_records}

    planner = TypoAttackPlanner(
        som_image_folder=som_image_folder,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        planner_model=args.planner_model,
        planner_api_key=args.planner_api_key,
        planner_base_url=args.planner_base_url,
        load_diffuser=False,
    )

    records = list(existing_records)
    for data in questions:
        question_id = data["question_id"]
        if question_id in completed_question_ids:
            continue

        image_name = data["image"]
        image_path = os.path.join(args.image_folder, image_name)
        record = planner.plan_attack(
            image_path=image_path,
            question=data["text"],
            correct_answer=data["answer"],
        )
        record["question_id"] = question_id
        record["image_original"] = image_name
        record["image_save_name"] = build_image_save_name(args.dataset, image_name, question_id)
        records.append(record)

        payload = {
            "meta": {
                "dataset": args.dataset,
                "image_folder": args.image_folder,
                "question_file": args.question_file,
                "som_image_folder": som_image_folder,
                "planner_model": args.planner_model,
                "planner_base_url": args.planner_base_url,
                "seed": args.seed,
                "slider": args.slider,
                "filter": args.filter,
            },
            "records": records,
        }
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"[plan] saved question_id={question_id}")
