import json
import re
from types import SimpleNamespace

from openai import OpenAI


class Message:
    def __init__(self):
        # Initialize the message list
        self.messages = []

    def add_system_message(self, instruction):
        """
        Add a system message with the given instruction.
        """
        self.messages.append({"role": "system", "content": instruction})

    def add_user_message(
        self, text=None, base64_images=None, detail="auto", image_first=False
    ):
        """
        Add a user message, which can include both text and multiple images. The images can appear before or after the text.

        Parameters:
        - text: The text content for the user message.
        - base64_images: A list of base64-encoded images.
        - detail: Image detail level (default is "auto").
        - image_first: If True, images will appear before text; if False, images will appear after text.
        """
        content_list = []

        # Append images first if image_first is True
        if base64_images and image_first:
            for i, base64_image in enumerate(base64_images):
                content_list.append({"type": "text", "text": f"Image {i}"})
                content_list.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": detail,
                        },
                    }
                )

        # Append text
        if text:
            content_list.append({"type": "text", "text": text})

        # Append images after text if image_first is False
        if base64_images and not image_first:
            for i, base64_image in enumerate(base64_images):
                content_list.append({"type": "text", "text": f"Image {i}"})
                content_list.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": detail,
                        },
                    }
                )

        if content_list:
            self.messages.append({"role": "user", "content": content_list})

    def add_user_message_test(
        self, text=None, base64_images=None, detail="auto", image_first=False
    ):
        """
        Add a user message, which can include both text and multiple images. The images can appear before or after the text.

        Parameters:
        - text: The text content for the user message.
        - base64_images: A list of base64-encoded images.
        - detail: Image detail level (default is "auto").
        - image_first: If True, images will appear before text; if False, images will appear after text.
        """
        content_list = []

        # Append images first if image_first is True
        if base64_images and image_first:
            for i, base64_image in enumerate(base64_images):
                content_list.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": detail,
                        },
                    }
                )

        # Append text
        if text:
            content_list.append({"type": "text", "text": text})

        # Append images after text if image_first is False
        if base64_images and not image_first:
            for i, base64_image in enumerate(base64_images):
                content_list.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": detail,
                        },
                    }
                )

        if content_list:
            self.messages.append({"role": "user", "content": content_list})

    def add_assistant_message(self, text):
        """
        Add an assistant message with the given text.
        """
        self.messages.append({"role": "assistant", "content": text})

    def get_messages(self):
        """
        Return the complete message log.
        """
        return self.messages


class CompletionRequest:
    def __init__(
        self,
        model,
        temperature=0.5,
        max_tokens=4095,
        top_p=0.5,
        response_format=None,
        api_key=None,
        base_url=None,
    ):
        """
        Initialize the completion request with model parameters.

        Parameters:
        - model: The model to be used for the completion (e.g., "gpt-4o-2024-08-06").
        - temperature: Sampling temperature (default: 0.5).
        - max_tokens: Maximum number of tokens to generate (default: 4095).
        - top_p: Nucleus sampling parameter (default: 0.5).
        - response_format: Expected response format (default: "Plan").
        """
        client_kwargs = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        if response_format is None:
            self.response_format = {"type": "text"}
        else:
            self.response_format = response_format
        self.message_handler = Message()

    @staticmethod
    def _extract_json_object(text):
        if text is None:
            raise ValueError("Model returned empty content.")

        text = text.strip()
        fenced_match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text)
        if fenced_match:
            return fenced_match.group(1)

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError(f"Unable to extract JSON object from response: {text}")
        return text[start : end + 1]

    def _build_wrapped_response(self, parsed, raw_content):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        parsed=parsed,
                        content=raw_content,
                    )
                )
            ]
        )

    def _json_schema_fallback(self):
        schema = self.response_format.model_json_schema()
        fallback_messages = list(self.message_handler.get_messages())
        fallback_messages.append(
            {
                "role": "system",
                "content": (
                    "Return ONLY one valid JSON object. Do not add markdown fences or extra text. "
                    f"The JSON object must satisfy this schema: {json.dumps(schema, ensure_ascii=True)}"
                ),
            }
        )
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=fallback_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            response_format={"type": "json_object"},
        )
        raw_content = completion.choices[0].message.content
        json_content = self._extract_json_object(raw_content)
        parsed = self.response_format.model_validate_json(json_content)
        return self._build_wrapped_response(parsed, raw_content)

    def set_response_format(self, response_format):
        """
        Set the response format for the completion request.
        """
        self.response_format = response_format

    def set_system_instruction(self, instruction):
        """
        Set the system instruction for the completion request.
        """
        self.message_handler.add_system_message(instruction)

    def add_user_message(
        self, text=None, base64_image=None, detail="auto", image_first=False
    ):
        """
        Add a user message to the completion request.
        """
        self.message_handler.add_user_message(text, base64_image, detail, image_first)

    def add_user_message_test(
        self, text=None, base64_image=None, detail="auto", image_first=False
    ):
        """
        Add a user message to the completion request.
        """
        self.message_handler.add_user_message_test(
            text, base64_image, detail, image_first
        )

    def add_assistant_message(self, text):
        """
        Add an assistant message to the completion request.
        """
        self.message_handler.add_assistant_message(text)

    def get_completion_payload(self):
        """
        Return the payload needed for the completion request, including the messages and model settings.
        """
        if self.response_format == {"type": "text"}:
            return self.client.chat.completions.create(
                model=self.model,
                messages=self.message_handler.get_messages(),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
        else:
            try:
                return self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=self.message_handler.get_messages(),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    response_format=self.response_format,
                )
            except Exception:
                # Some OpenAI-compatible endpoints do not implement `parse`,
                # but can still return JSON compatible with the same schema.
                return self._json_schema_fallback()
