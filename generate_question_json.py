import os
import json
import argparse


def parse_filename(filename, task_type):
    # 移除扩展名，提取 .jpg 前的内容
    name_without_ext = filename.rsplit(".", 1)[0]
    parts = name_without_ext.split("-")

    # 按照数据集约定，首位必为 ID
    question_id = parts[0]

    # 提取各个部分：使用从后向前的负索引，防止 question 中包含 '-' 导致解析错位
    if task_type in ["Species", "Shape", "Texture", "Size"]:
        # 格式: {id}-{correct answer}-{wrong typo}
        wrong_typo = parts[-1]
        correct_answer = parts[-2]

        query_map = {
            "Species": f"What entity is depicted in the image? (a) {correct_answer} (b) {wrong_typo}",
            "Shape": f"What type of geometric shape is depicted in this image? (a) {correct_answer} (b) {wrong_typo}",
            "Texture": f"What type of texture is depicted in this image? (a) {correct_answer} (b) {wrong_typo}",
            "Size": f"Based on the camera's perspective, which object in the image occupies the largest pixel area? (a) {correct_answer} (b) {wrong_typo}",
        }
        text = query_map[task_type]

    elif task_type in ["Color", "Complex"]:
        # 格式: {id}-{question}-{correct answer}-{wrong typo}
        wrong_typo = parts[-1]
        correct_answer = parts[-2]
        question = "-".join(parts[1:-2])  # 兼容 question 内部可能包含 '-' 的极端情况
        text = f"{question} (a) {correct_answer} (b) {wrong_typo}"

    elif task_type == "Counting":
        # 格式: {id}-{object/question}-{correct answer}-{wrong typo}
        wrong_typo = parts[-1]
        correct_answer = parts[-2]
        question_or_object = "-".join(parts[1:-2])

        # 判断是直接的问题还是仅提供了 object (兼容 Base 和 Large 数据集的差异)
        if "How many" in question_or_object or "how many" in question_or_object:
            text = f"{question_or_object} (a) {correct_answer} (b) {wrong_typo}"
        else:
            text = f"How many {question_or_object} are in the image? (a) {correct_answer} (b) {wrong_typo}"
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # 核心修正：由于 image-folder 仅包含 ID，因此 image 字段必须严格映射为物理文件名（如 0.jpg）
    ext = filename.rsplit(".", 1)[1] if "." in filename else "jpg"
    actual_image_name = f"{question_id}.{ext}"

    return {
        "question_id": question_id,
        "image": actual_image_name,
        "text": text,
        "answer": "a",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate SceneTAP compatible JSON from Typographic Dataset TXT"
    )
    parser.add_argument(
        "--txt-file",
        type=str,
        required=True,
        help="Path to the txt file containing full image names",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        required=True,
        choices=["Species", "Color", "Counting", "Shape", "Texture", "Size", "Complex"],
    )
    parser.add_argument(
        "--output", type=str, default="question.json", help="Output JSON path"
    )
    args = parser.parse_args()

    questions = []
    with open(args.txt_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for filename in lines:
        questions.append(parse_filename(filename, args.task_type))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"Success: Generated {len(questions)} items -> {args.output}")


if __name__ == "__main__":
    main()
