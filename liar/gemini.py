"""
What is this file?
This file manages Gemini-based explanation generation for LIAR prediction results.

What is its responsibility?
It creates a grounded prompt using the user's statement, the selected model prediction, retrieved similar statements, and asks Gemini to generate a short explanation.
"""

import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


def prompt_creator(
    input_data: dict,
    prediction_result: dict,
    similar_statements: list[dict],
    max_words: int = 120,
) -> list:
    """
    Create a prompt for Gemini based on prediction result and retrieved similar statements.
    """

    system_msg = """
You are a political fact-checking assistant.

Your task is to explain a machine learning prediction using only:
- the user's statement,
- the selected model's prediction result,
- the retrieved similar statements from the training dataset.

Rules:
- Do not invent external facts.
- Do not claim the prediction is objectively true.
- Do not fact-check the real world.
- Explain patterns, similarities, uncertainty, and possible rhetorical signals.
- Only three labels are possible: unreliable, questionable, trustworthy.
- Write a complete answer.
- Do not start an unfinished quote.
- Do not end mid-sentence.
"""

    statement = input_data.get("statement", "unknown")
    speaker = input_data.get("speaker", "unknown")
    context = input_data.get("context", "unknown")

    model_name = prediction_result.get("model_name", "unknown")
    prediction = prediction_result.get("prediction", "unknown")
    confidence = prediction_result.get("confidence", 0)
    class_probabilities = prediction_result.get("class_probabilities", {})

    similar_text = "\n\n".join(
        [
            f"Example {index + 1}:\n"
            f"Label: {item.get('label', 'unknown')}\n"
            f"Speaker: {item.get('speaker', 'unknown')}\n"
            f"Context: {item.get('context', 'unknown')}\n"
            f"Statement: {item.get('statement', 'unknown')}"
            for index, item in enumerate(similar_statements)
        ]
    )

    human_msg = f"""
User statement:
{statement}

Speaker:
{speaker}

Context:
{context}

Selected model:
{model_name}

Model prediction:
{prediction}

Model confidence:
{confidence:.2%}

Class probabilities:
{class_probabilities}

Retrieved similar statements:
{similar_text}

Write one complete explanation in maximum {max_words} words.

The explanation must cover:
1. The dominant pattern in the retrieved examples.
2. How that pattern relates to the model prediction.
3. What uncertainty remains.

Important:
- Do not use markdown tables.
- Do not end mid-sentence.
- End with a complete sentence.
"""

    return [
        SystemMessage(content=system_msg),
        HumanMessage(content=human_msg),
    ]


def build_fallback_explanation(
    prediction_result: dict,
    similar_statements: list[dict],
) -> str:
    """
    Build a deterministic fallback explanation if Gemini is unavailable or returns an incomplete answer.
    """

    prediction = prediction_result.get("prediction", "unknown")
    confidence = prediction_result.get("confidence", 0)
    model_name = prediction_result.get("model_name", "unknown")

    labels = [
        item.get("label", "unknown")
        for item in similar_statements
    ]

    label_counts = {
        label: labels.count(label)
        for label in sorted(set(labels))
    }

    return (
        f"The selected model ({model_name}) predicted '{prediction}' "
        f"with {confidence:.2%} confidence. "
        f"The retrieved similar statements show this label distribution: {label_counts}. "
        "This suggests the result should be interpreted as pattern-based support, "
        "not as an objective fact-check. The retrieved examples provide useful context, "
        "but the final prediction still carries uncertainty."
    )


def generate_gemini_explanation(
    input_data: dict,
    prediction_result: dict,
    similar_statements: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 600,
    max_words: int = 120,
    max_retries: int = 3,
    retry_delay_seconds: int = 5,
    min_valid_length: int = 120,
) -> str:
    """
    Generate a Gemini explanation for a prediction result.
    If Gemini is temporarily unavailable or returns an incomplete answer, retry before returning a clean fallback explanation.
    """

    messages = prompt_creator(
        input_data=input_data,
        prediction_result=prediction_result,
        similar_statements=similar_statements,
        max_words=max_words,
    )

    llm_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
        max_tokens=max_tokens,
    )

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = llm_model.invoke(messages)
            explanation = response.text.strip()

            is_complete_sentence = explanation.endswith((".", "!", "?"))

            if len(explanation) >= min_valid_length and is_complete_sentence:
                return explanation

            last_error = (
                f"Gemini returned an incomplete explanation: {repr(explanation)}"
            )

            print(f"Gemini explanation incomplete on attempt {attempt}: {last_error}")

        except Exception as error:
            last_error = error
            print(f"Gemini explanation failed on attempt {attempt}: {error}")

        if attempt < max_retries:
            time.sleep(retry_delay_seconds)

    fallback = build_fallback_explanation(
        prediction_result=prediction_result,
        similar_statements=similar_statements,
    )

    print(
        "Gemini explanation was not used because the response was unavailable "
        f"or incomplete. Last error: {last_error}"
    )

    return fallback
