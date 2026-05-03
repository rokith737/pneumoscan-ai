"""
LLM Report Generator — OpenAI GPT-4
Generates a plain-English radiology-style summary from model predictions.
"""

from __future__ import annotations

import os
from openai import AsyncOpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """You are an expert radiologist AI assistant.
Your role is to generate a clear, concise, plain-English clinical summary
based on a deep-learning model's chest X-ray analysis result.

Guidelines:
- Write in 3 short paragraphs: (1) Findings, (2) Interpretation, (3) Recommendation
- Use plain English — avoid heavy jargon, but keep clinical accuracy
- Be appropriately cautious: always recommend professional radiologist review
- Never claim to replace a licensed physician
- Keep total response under 200 words
"""

PNEUMONIA_TEMPLATE = """
The deep-learning model analysed the uploaded chest X-ray and produced the following:
- Prediction : {prediction}
- Confidence : {confidence_pct:.1f}%
- P(Normal)  : {p_normal:.1f}%
- P(Pneumonia): {p_pneumonia:.1f}%

Please generate a clinical summary report based on these results.
"""


async def generate_report(
    prediction: str,
    confidence: float,
    probabilities: dict[str, float],
) -> str:
    """Call GPT-4 and return a formatted diagnostic summary."""

    if not OPENAI_API_KEY:
        return _fallback_report(prediction, confidence, probabilities)

    user_msg = PNEUMONIA_TEMPLATE.format(
        prediction=prediction,
        confidence_pct=confidence * 100,
        p_normal=probabilities.get("NORMAL", 0) * 100,
        p_pneumonia=probabilities.get("PNEUMONIA", 0) * 100,
    )

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=300,
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()


def _fallback_report(
    prediction: str,
    confidence: float,
    probabilities: dict[str, float],
) -> str:
    """Rule-based fallback when no API key is present."""
    p_pneu = probabilities.get("PNEUMONIA", 0) * 100
    p_norm = probabilities.get("NORMAL", 0) * 100
    conf   = confidence * 100

    if prediction == "PNEUMONIA":
        finding = (
            f"The AI model detected radiological patterns consistent with "
            f"pneumonia with a confidence of {conf:.1f}% "
            f"(Pneumonia probability: {p_pneu:.1f}%, Normal: {p_norm:.1f}%)."
        )
        interpretation = (
            "Infiltrates or consolidation patterns typically associated with "
            "pneumonia may be present in the image."
        )
        recommendation = (
            "Urgent clinical correlation is advised. A qualified radiologist "
            "should review the original image. If symptoms are present, "
            "consider further workup including laboratory tests."
        )
    else:
        finding = (
            f"The AI model found no clear signs of pneumonia "
            f"(Normal probability: {p_norm:.1f}%, Pneumonia: {p_pneu:.1f}%; "
            f"confidence: {conf:.1f}%)."
        )
        interpretation = (
            "The chest X-ray appears within normal radiological parameters "
            "based on the model's analysis."
        )
        recommendation = (
            "Routine follow-up as clinically indicated. This result does not "
            "replace a formal radiologist report — please consult a licensed "
            "physician for any clinical decision."
        )

    return f"{finding}\n\n{interpretation}\n\n{recommendation}"
