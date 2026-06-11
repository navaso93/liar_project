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
    max_words: int = 200,
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
- Do not use markdown tables.
- Do not use bold text.
- Do not use bullet points.
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

Write one complete explanation in 4 to 6 sentences and maximum {max_words} words.
Do not write only one sentence.

Your explanation must cover:
1. What pattern appears in the retrieved similar statements.
2. How that pattern supports or weakens the selected model prediction.
3. What uncertainty remains.

Important:
- This is not an objective fact-check.
- This is an explanation of model behavior and retrieved examples.
- Do not simply repeat the label distribution.
- Explain the retrieved examples specifically.
- Mention at least two retrieved examples by their speaker or label.
- Write 4 to 6 complete sentences.
- Do not use bullet points.
- Do not use markdown formatting.
- Do not use bold text.
- Do not end mid-sentence.
- Use natural language.
"""

    return [
        SystemMessage(content=system_msg),
        HumanMessage(content=human_msg),
    ]


def extract_response_text(response) -> str:
    """
    Extract plain text from a Gemini/LangChain response object.
    """

    explanation = getattr(response, "content", None)

    if isinstance(explanation, list):
        explanation = " ".join(
            [
                item.get("text", "")
                if isinstance(item, dict)
                else str(item)
                for item in explanation
            ]
        )

    if not explanation:
        explanation = getattr(response, "text", "")

    return str(explanation).strip()


def is_valid_gemini_explanation(explanation: str) -> bool:
    """
    Check whether the Gemini explanation is complete enough to show to the user.
    """

    if not explanation:
        return False

    sentence_count = (
        explanation.count(".")
        + explanation.count("!")
        + explanation.count("?")
    )

    ends_cleanly = explanation.endswith((".", "!", "?"))
    is_long_enough = len(explanation) >= 250

    mentions_retrieved_examples = (
        "retrieved" in explanation.lower()
        or "similar" in explanation.lower()
        or "example" in explanation.lower()
        or "statement" in explanation.lower()
    )

    return (
        is_long_enough
        and sentence_count >= 3
        and ends_cleanly
        and mentions_retrieved_examples
    )


def build_retry_prompt(previous_answer: str) -> HumanMessage:
    """
    Build a corrective prompt when Gemini returns a weak or incomplete answer.
    """

    return HumanMessage(
        content=f"""
Your previous answer was incomplete, too short, or too shallow:

{previous_answer}

Rewrite it as a complete 4 to 6 sentence explanation.
Mention at least two retrieved examples by speaker or label.
Explain how the retrieved examples relate to the model prediction.
Explain what uncertainty remains.
End with a complete sentence.
Do not use markdown formatting.
Do not use bold text.
Do not use bullet points.
"""
    )


def build_fallback_explanation(
    prediction_result: dict,
    similar_statements: list[dict],
) -> str:
    """
    Build a deterministic fallback explanation if Gemini is unavailable or repeatedly returns weak answers.
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
    temperature: float = 0.3,
    max_tokens: int = 1200,
    max_words: int = 200,
    max_retries: int = 3,
    retry_delay_seconds: int = 1,
) -> str:
    """
    Generate a Gemini explanation for a prediction result.
    Return Gemini output only when it is complete and useful.
    Use fallback when Gemini fails or repeatedly returns weak/incomplete text.
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
    last_incomplete_answer = None

    for attempt in range(1, max_retries + 1):
        try:
            response = llm_model.invoke(messages)
            explanation = extract_response_text(response)

            if is_valid_gemini_explanation(explanation):
                return explanation

            last_incomplete_answer = explanation
            last_error = (
                "Gemini returned a weak or incomplete explanation: "
                f"{repr(explanation)}"
            )

            print(f"Gemini explanation rejected on attempt {attempt}: {last_error}")

            messages = prompt_creator(
                input_data=input_data,
                prediction_result=prediction_result,
                similar_statements=similar_statements,
                max_words=max_words,
            )

            messages.append(
                build_retry_prompt(
                    previous_answer=last_incomplete_answer,
                )
            )

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
        "Gemini explanation was not used because Gemini failed or repeatedly returned weak text. "
        f"Last error: {last_error}"
    )

    return fallback
