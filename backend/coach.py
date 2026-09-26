MODEL = "llama3.1:8b"

SYSTEM_PROMPT = (
    "You are a supportive fitness coach giving spoken feedback during a bodyweight workout. "
    "You will be given a structured verdict that has ALREADY been decided by a separate "
    "analysis system. Your only job is to phrase that verdict as one short, encouraging "
    "sentence the user hears between reps. "
    "Rules: "
    "Do NOT add, change, or second-guess the verdict. "
    "Do NOT invent angles, numbers, or faults that are not in the verdict. "
    "Do NOT diagnose anything yourself. "
    "If corrections are listed, mention every one of them. "
    "If the verdict says the rep was good, simply encourage. "
    "Keep it under 20 words, warm and plain. Output only the sentence."
)

# --- Rule-based fault selection ----------------------------------------------
#
# A one-sentence cue tends to keep the first faults it is given and drop the
# rest, so the rule layer decides which faults are spoken: at most
# MAX_FAULTS_SURFACED, in a fixed priority. Posture comes first, then knee
# tracking, then depth, which affects effectiveness rather than safety. Only
# an explicit False is a fault; None means the criterion was not measured.
MAX_FAULTS_SURFACED = 2

# (verdict_key, exercise or None for any, canonical name, fallback cue,
#  description for the LLM)
_FAULTS = [
    ("trunk_ok",       None,    "trunk_lean",
     "keep your chest up",
     "Torso: leaned too far forward."),
    ("knee_travel_ok", "lunge", "knee_travel",
     "keep your front knee over your ankle",
     "Front knee: drifted too far past the toes."),
    ("depth_ok",       "squat", "shallow_depth",
     "squat a little deeper",
     "Depth: did not reach parallel."),
    ("depth_ok",       "lunge", "shallow_depth",
     "drop your front knee lower",
     "Depth: front knee did not bend enough."),
]


def _breached(verdict):
    """Every breached criterion, uncapped, in priority order."""
    exercise = verdict.get("exercise", "squat")
    found = []
    for key, exo, name, cue, desc in _FAULTS:
        if exo is not None and exo != exercise:
            continue
        if verdict.get(key) is False:  # None (unmeasured) is not a fault
            found.append((name, cue, desc))
    return found


def detect_faults(verdict):
    """Canonical names of every fault detected, including any beyond the cue
    cap. The interface renders these, so a fault that could not fit into one
    spoken sentence is still visible to the user."""
    return [name for name, _, _ in _breached(verdict)]


def select_faults(verdict):
    """The prioritised, capped (cue, description) faults for a verdict."""
    return [(cue, desc) for _, cue, desc in _breached(verdict)][:MAX_FAULTS_SURFACED]


def _label(verdict):
    return "Lunge rep" if verdict.get("exercise") == "lunge" else "Rep"


def _verdict_to_text(verdict):
    """Build the message the LLM phrases: only the selected faults, or a
    praise instruction when the rep is clean."""
    n = verdict["rep_number"]
    faults = select_faults(verdict)
    if not faults:
        return f"{_label(verdict)} {n} completed. All checks passed. Encourage the user briefly."
    parts = [f"{_label(verdict)} {n} completed."]
    parts += [desc for _, desc in faults]
    parts.append("Mention each correction above and no others.")
    return " ".join(parts)


async def phrase_feedback(verdict):
    """Phrase a decided verdict as one coaching sentence.

    Falls back to fixed wording if the model is unavailable."""
    user_msg = _verdict_to_text(verdict)
    try:
        from ollama import AsyncClient  # lazy import so pure logic is testable
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
    """Fixed coaching text for when the model is unavailable. Covers the same
    selected faults as the model path."""
    n = verdict["rep_number"]
    faults = select_faults(verdict)
    if not faults:
        return f"{_label(verdict)} {n}: great form, keep it up."
    return f"{_label(verdict)} {n}: " + " and ".join(cue for cue, _ in faults) + "."


async def warm_up():
    """Load the model at server start.

    A cold first call takes about 2.5 s against 0.7 to 0.9 s warm, so the cost
    is paid before the first repetition rather than on it.
    """
    try:
        from ollama import AsyncClient
        await AsyncClient().chat(
            model=MODEL,
            messages=[{"role": "user", "content": "Reply with: ready"}],
            options={"temperature": 0.0, "num_predict": 4},
        )
        print(f"{MODEL} warmed up")
    except Exception as e:
        print(f"Warm-up skipped, model unavailable: {e}")
