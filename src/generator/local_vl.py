"""Local vision-language answer generation using Qwen3-VL."""

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


SYSTEM_PROMPT = """You are a document analysis assistant. You are given a question about a document
and a relevant region (image) extracted from that document. Answer the question based solely on the
information visible in the provided image region.

If the image does not contain enough information to answer the question, say so clearly.
Be concise and factual."""


class LocalVLGenerator:
    """Generate answers using a local Qwen3-VL model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str | None = None,
        max_tokens: int = 1024,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_tokens = max_tokens
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.model.eval()

    def generate(self, query: str, region_image: Image.Image) -> str:
        """Generate an answer grounded in the retrieved region image."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": region_image},
                    {"type": "text", "text": f"Question: {query}"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output[0]

    def generate_with_context(
        self,
        query: str,
        region_images: list[Image.Image],
        region_labels: list[str] | None = None,
    ) -> str:
        """Generate an answer using multiple retrieved regions."""
        content = []
        for i, img in enumerate(region_images):
            label = region_labels[i] if region_labels else f"Region {i + 1}"
            content.append({"type": "text", "text": f"\n{label}:"})
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": f"\nQuestion: {query}"})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output[0]
