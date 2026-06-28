from ollama import AsyncClient

MODEL = "llama3.1:8b"

SYSTEM_PROMPT = (
    "You are a supportive fitness coach giving spoken feedback during a squat workout. "
    "You will be given a structured verdict that has ALREADY been decided by a separate "
    "analysis system. Your only job is to phrase that verdict as one short, encouraging "
    "sentence the user hears between reps. "
    "Rules: "
    "Do NOT add, change, or second-guess the verdict. "
    "Do NOT invent angles, numbers, or faults that are not in the verdict. "
    "Do NOT diagnose anything yourself. "
    "If the verdict says the rep was good, simply encourage. "
    "Keep it under 20 words, warm and plain. Output only the sentence."
)


def _verdict_to_text(verdict):
    parts = [f"Rep {verdict['rep_number']} completed."]
    if verdict["depth_ok"]:
        parts.append("Depth: full depth reached.")
    else:
        parts.append("Depth: did not reach parallel, needs to go deeper.")
    if verdict["trunk_ok"]:
        parts.append("Torso: stable and upright.")
    else:
        parts.append("Torso: leaned too far forward.")
    return " ".join(parts)


async def phrase_feedback(verdict):
    """Translate a decided verdict into one coaching sentence.
    Falls back to a deterministic string if the model is unavailable."""
    user_msg = _verdict_to_text(verdict)
    try:
        response = await AsyncClient().chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            options={"temperature": 0.4},
        )
        text = response["message"]["content"].strip()
        return text.strip('"').strip()
    except Exception as e:
        print(f"Ollama unavailable, using fallback: {e}")
        return _fallback(verdict)


def _fallback(verdict):
    if verdict["depth_ok"] and verdict["trunk_ok"]:
        return f"Rep {verdict['rep_number']}: great form, keep it up."
    tips = []
    if not verdict["depth_ok"]:
        tips.append("squat a little deeper")
    if not verdict["trunk_ok"]:
        tips.append("keep your chest up")
    return f"Rep {verdict['rep_number']}: " + " and ".join(tips) + "."
