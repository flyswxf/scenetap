import base64
import os
import time
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Function to encode the image
from pydantic import BaseModel

from utils.get_rectangle_by_mask import largest_inscribed_rectangle

# from utils.som import SoM
from utils.completion_request import CompletionRequest


from utils.text_diffuser import TextDiffuser


class PlanSom(BaseModel):
    image_analysis: str
    correct_answer: str
    incorrect_answer: str
    adversarial_text: str
    text_position_number: int
    text_placement: str
    short_caption_with_adversarial_text: str


class PlanSomAdjust(BaseModel):
    adjust_explanation: str
    adjust_plan: PlanSom


def pil_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def format_instance_json(instance):
    # Get the attribute names from the class definition
    attributes = instance.__class__.__annotations__.keys()

    # Retrieve the values from the instance
    values = {attr: getattr(instance, attr) for attr in attributes}

    return values


def find_text_region(
    text,
    left,
    top,
    right,
    bottom,
    font_path="./fonts/arial.ttf",
    font_size=20,
    aspect_ratio_threshold=0.1,
):
    # Load the font (you may need to provide the correct font path)
    font = ImageFont.truetype(font_path, font_size)

    # Calculate the width and height of the original region
    w = right - left
    h = bottom - top

    # Get the text size (width and height)
    text_width, text_height = font.getsize(text)

    # Calculate text aspect ratio
    text_aspect_ratio = text_height / text_width

    # Calculate the region aspect ratio
    region_aspect_ratio = h / w

    # Compare the two aspect ratios
    aspect_ratio_difference = abs(region_aspect_ratio - text_aspect_ratio)

    if aspect_ratio_difference > aspect_ratio_threshold:
        # If the aspect ratios differ too much, adjust the region
        if text_aspect_ratio > region_aspect_ratio:
            # Text is taller relative to the region aspect ratio, adjust height
            scaled_height = h
            scaled_width = scaled_height / text_aspect_ratio
        else:
            # Text is wider relative to the region aspect ratio, adjust width
            scaled_width = w
            scaled_height = scaled_width * text_aspect_ratio

        # Center the found region within the original [left, top, right, bottom]
        find_left = left + (w - scaled_width) / 2
        find_top = top + (h - scaled_height) / 2
        find_right = find_left + scaled_width
        find_bottom = find_top + scaled_height

        return int(find_left), int(find_top), int(find_right), int(find_bottom)

    # If aspect ratio is close enough, return the original region
    return int(left), int(top), int(right), int(bottom)


class TypoAttackPlanner:
    def __init__(
        self,
        som_image_folder=None,
        temperature=0.2,
        max_tokens=4095,
        top_p=0.1,
        planner_model="gpt-4o-2024-08-06",
        planner_api_key=None,
        planner_base_url=None,
        load_diffuser=True,
    ):
        """
        Initialize the TypoAttackPlanner class.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        self.som_image_folder = som_image_folder
        self.planner_model = planner_model
        self.planner_api_key = planner_api_key
        self.planner_base_url = planner_base_url

        self.diffuser = TextDiffuser() if load_diffuser else None

        # system instruction
        with open("prompt/attack_step_give_answer_combine.txt", "r") as file:
            self.instruction_combine = file.read()

        with open("prompt/attack_adjust_plan.txt", "r") as file:
            self.instruction_adjust_plan = file.read()

    def _ensure_diffuser(self):
        if self.diffuser is None:
            self.diffuser = TextDiffuser()

    def _load_som_assets(self, image_path):
        if self.som_image_folder is None:
            raise ValueError("`som_image_folder` must be provided to load SoM assets.")

        image_name = os.path.basename(image_path)
        seg_image = Image.open(os.path.join(self.som_image_folder, image_name)).convert(
            "RGB"
        )
        mask = np.load(
            os.path.join(self.som_image_folder, image_name.replace(".jpg", ".npy")),
            allow_pickle=True,
        )
        return image_name, seg_image, mask

    @staticmethod
    def _plan_to_dict(plan_detail):
        if isinstance(plan_detail, BaseModel):
            return plan_detail.model_dump()
        return plan_detail

    @staticmethod
    def _plan_from_dict(plan_detail):
        if isinstance(plan_detail, BaseModel):
            return plan_detail
        return PlanSom.model_validate(plan_detail)

    def _request_attack_plan(self, image, seg_image, question, correct_answer):
        base64_image = pil_to_base64(image)
        base64_image_som = pil_to_base64(seg_image)

        completion_request = CompletionRequest(
            model=self.planner_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            response_format=PlanSom,
            api_key=self.planner_api_key,
            base_url=self.planner_base_url,
        )
        completion_request.set_system_instruction(self.instruction_combine)
        user_text = (
            "Image 0 is the original image, image 1 is the corresponding segmentation map. "
            f"Observe the image and the corresponding segmentation map carefully. Question to attack: {question}. "
            f"Correct answer: {correct_answer}. Please provide a detailed, step-by-step plan for achieving this goal."
        )
        completion_request.add_user_message(
            text=user_text,
            base64_image=[base64_image, base64_image_som],
            image_first=True,
        )
        completion = completion_request.get_completion_payload()
        plan_detail = completion.choices[0].message.parsed

        print("plan_detail:")
        print("image_analysis:", plan_detail.image_analysis)
        print("correct_answer:", plan_detail.correct_answer)
        print("incorrect_answer:", plan_detail.incorrect_answer)
        print("adversarial_text:", plan_detail.adversarial_text)
        print("text_placement:", plan_detail.text_placement)
        print("text_position_number:", plan_detail.text_position_number)
        print(
            "short_caption_with_adversarial_text:",
            plan_detail.short_caption_with_adversarial_text,
        )

        completion_request.add_assistant_message(text=f"{plan_detail}")

        completion_request.set_response_format(PlanSomAdjust)
        completion_request.add_user_message(text=self.instruction_adjust_plan)
        completion = completion_request.get_completion_payload()
        plan_adjust = completion.choices[0].message.parsed

        plan_detail_origin = plan_detail.copy()
        plan_detail = plan_adjust.adjust_plan
        explanation = plan_adjust.adjust_explanation
        print("explanation:", explanation)
        print("plan_detail:")
        print("image_analysis:", plan_detail.image_analysis)
        print("correct_answer:", plan_detail.correct_answer)
        print("incorrect_answer:", plan_detail.incorrect_answer)
        print("adversarial_text:", plan_detail.adversarial_text)
        print("text_placement:", plan_detail.text_placement)
        print("text_position_number:", plan_detail.text_position_number)
        print(
            "short_caption_with_adversarial_text:",
            plan_detail.short_caption_with_adversarial_text,
        )

        return plan_detail_origin, plan_detail, explanation

    def _compute_text_region(self, image, mask, plan_detail):
        if int(plan_detail.text_position_number) <= len(mask):
            target_mask_index = int(plan_detail.text_position_number) - 1
            target_mask = mask[target_mask_index]["segmentation"]
        else:
            print("text_position_number is out of range, use the largest mask")
            target_mask_index = 0
            target_mask = mask[0]["segmentation"]

        label = True
        x, y, w, h = largest_inscribed_rectangle(target_mask, label)
        print("rectangle [x, y, w, h]:", [x, y, w, h])
        mask_width, mask_height = target_mask.T.shape

        left, top, right, bottom = (
            x / mask_width * image.width,
            y / mask_height * image.height,
            (x + w) / mask_width * image.width,
            (y + h) / mask_height * image.height,
        )

        print(
            "rectangle [(left, top), (right, bottom)]:",
            [(int(left), int(top)), (int(right), int(bottom))],
        )

        left, top, right, bottom = find_text_region(
            plan_detail.adversarial_text,
            left,
            top,
            right,
            bottom,
            font_path="./fonts/arial.ttf",
            font_size=20,
            aspect_ratio_threshold=0.1,
        )
        print(
            "Resized rectangle [(left, top), (right, bottom)]:",
            [(int(left), int(top)), (int(right), int(bottom))],
        )
        return {
            "mask_index": int(target_mask_index),
            "left": int(left),
            "top": int(top),
            "right": int(right),
            "bottom": int(bottom),
        }

    def plan_attack(self, image_path, question, correct_answer):
        image = Image.open(image_path).convert("RGB")
        image_name, seg_image, mask = self._load_som_assets(image_path)
        plan_detail_origin, plan_detail, explanation = self._request_attack_plan(
            image=image,
            seg_image=seg_image,
            question=question,
            correct_answer=correct_answer,
        )
        rectangle = self._compute_text_region(
            image=image, mask=mask, plan_detail=plan_detail
        )
        return {
            "image": image_name,
            "question": question,
            "correct_answer": correct_answer,
            "planner_model": self.planner_model,
            "planner_explanation": explanation,
            "plan_detail_origin": self._plan_to_dict(plan_detail_origin),
            "plan_detail": self._plan_to_dict(plan_detail),
            "rectangle": rectangle,
        }

    def render_from_plan(
        self,
        image_path,
        plan_detail,
        rectangle,
        scale_factor=2,
        regional_diffusion=True,
    ):
        self._ensure_diffuser()
        plan_detail = self._plan_from_dict(plan_detail)
        image = Image.open(image_path).convert("RGB")
        two_point_positions = [
            (int(rectangle["left"]), int(rectangle["top"])),
            (int(rectangle["right"]), int(rectangle["bottom"])),
        ]
        diffusion_result = self.diffuser.generate(
            two_point_positions,
            image_path,
            plan_detail.adversarial_text,
            plan_detail.short_caption_with_adversarial_text,
            radio="Two Points",
            scale_factor=scale_factor,
            regional_diffusion=regional_diffusion,
        )

        diffusion_images = diffusion_result[0]
        diffusion_images = [
            diffusion_image.resize((image.width, image.height))
            for diffusion_image in diffusion_images
        ]
        return diffusion_images

    def attack(self, image_path, question, correct_answer):
        """
        Applies a 'typo attack' on the input PIL image and returns the modified image.

        Returns:
        The modified image with the applied 'typo attack'.
        """
        plan_record = self.plan_attack(
            image_path=image_path, question=question, correct_answer=correct_answer
        )
        _, seg_image, _ = self._load_som_assets(image_path)
        diffusion_images = self.render_from_plan(
            image_path=image_path,
            plan_detail=plan_record["plan_detail"],
            rectangle=plan_record["rectangle"],
        )
        return (
            diffusion_images,
            seg_image,
            PlanSom.model_validate(plan_record["plan_detail_origin"]),
            PlanSom.model_validate(plan_record["plan_detail"]),
        )
