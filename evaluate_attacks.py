import argparse
import json
import logging
import os

from PIL import Image

from utils.completion_request import CompletionRequest
from utils.lingo_judge import LingoJudge
from utils.typo_attack_planner import pil_to_base64
from utils.utils import is_correct_answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-file", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_tokens", type=int, default=4095)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--log_dir", type=str, default="./log")
    args = parser.parse_args()

    with open(args.render_file, "r", encoding="utf-8") as f:
        render_payload = json.load(f)

    meta = render_payload["meta"]
    records = render_payload["records"]
    dataset = meta["dataset"]

    log_dir = os.path.join(args.log_dir, args.model, dataset, "SceneTAP")
    log_dir = os.path.join(log_dir, f"slider_{meta['slider']}", f"filter_{meta['filter']}")
    log_dir = os.path.join(log_dir, f"seed_{meta['seed']}")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("evaluate_attacks")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    file_handler = logging.FileHandler(os.path.join(log_dir, "log.txt"), "a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(""))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(""))
    logger.addHandler(stream_handler)

    answers_file = os.path.join(log_dir, "answer.jsonl")
    if dataset == "LingoQA":
        lingo_judge = LingoJudge()
    else:
        lingo_judge = None

    ans_file_list = []
    correct = 0
    total = 0
    for record in records:
        question = record["question"]
        correct_answer = record["correct_answer"]
        output_list = []
        judge_list = []

        for image_path in record["diffusion_image_paths"]:
            image = Image.open(image_path).convert("RGB")
            base64_image = pil_to_base64(image)
            completion_request = CompletionRequest(
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                api_key=args.api_key,
                base_url=args.base_url,
            )
            completion_request.add_user_message_test(
                text=question,
                base64_image=[base64_image],
                image_first=True,
                detail="auto",
            )
            completion = completion_request.get_completion_payload()
            answer = completion.choices[0].message.content.lower()
            output_list.append(answer)
            judge_list.append(is_correct_answer(answer, correct_answer, question, dataset, lingo_judge=lingo_judge))

        is_correct = not (False in judge_list)
        if is_correct:
            correct += 1

        log_data = {
            "question_id": record["question_id"],
            "image": record["image_original"],
            "text": question,
            "outputs": output_list,
            "answer": correct_answer,
            "plan_detail_origin": record["plan_detail_origin"],
            "plan_detail": record["plan_detail"],
            "judge_list": judge_list,
            "is_correct": is_correct,
            "attack_image_path": record["attack_image_path"],
            "diffusion_image_paths": record["diffusion_image_paths"],
        }
        ans_file_list.append(log_data)
        logger.info(log_data)
        total += 1

    with open(answers_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(ans_file_list, indent=2, ensure_ascii=False))

    logger.info(f"Correct: {correct}/{total}")
    logger.info(f"Accuracy: {correct / total}")
    logger.info(f"ASR: {(total - correct) / total}")
