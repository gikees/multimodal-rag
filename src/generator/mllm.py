"""MLLM answer generation using Google Gemini Flash."""

import base64
import io

import google.generativeai as genai
from PIL import Image


SYSTEM_PROMPT = """You are a document analysis assistant. You are given a question about a document
and a relevant region (image) extracted from that document. Answer the question based solely on the
information visible in the provided image region.

If the image does not contain enough information to answer the question, say so clearly.
Be concise and factual."""


def encode_image_to_base64(image: Image.Image) -> str:
    """Encode a PIL Image to a base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class GeminiGenerator:
    """Generate answers using Gemini Flash with retrieved document regions."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        query: str,
        region_image: Image.Image,
    ) -> str:
        """Generate an answer grounded in the retrieved region image.

        Args:
            query: User's question.
            region_image: Cropped region image from the document.

        Returns:
            Generated answer string.
        """
        response = self.model.generate_content(
            [
                SYSTEM_PROMPT,
                region_image,
                f"Question: {query}",
            ],
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        return response.text

    def generate_with_context(
        self,
        query: str,
        region_images: list[Image.Image],
        region_labels: list[str] | None = None,
    ) -> str:
        """Generate an answer using multiple retrieved regions.

        Args:
            query: User's question.
            region_images: List of cropped region images.
            region_labels: Optional labels for each region.

        Returns:
            Generated answer string.
        """
        content = [SYSTEM_PROMPT]
        for i, img in enumerate(region_images):
            label = region_labels[i] if region_labels else f"Region {i + 1}"
            content.append(f"\n{label}:")
            content.append(img)
        content.append(f"\nQuestion: {query}")

        response = self.model.generate_content(
            content,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        return response.text
