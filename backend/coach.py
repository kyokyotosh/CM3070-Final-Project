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

# --- Deterministic fault selection -----------------------------------------
#
# When several faults occur in one rep, a single-sentence coaching cue used to
# drop whichever fault was listed last: an LLM compressing to one sentence
# keeps the first items and loses the tail. Coverage is restored here
# deterministically. The form layer, not the language model, decides which
# faults to surface, in a fixed priority, and caps how many are passed for
# phrasing so the whole set survives compression into one sentence.
#
# Priority rationale: posture (spinal) cues first, then joint-tracking cues,
# then depth, which is an effectiveness cue rather than a safety one. This
# ordering is a coaching design choice and can be re-ordered without touching
# the phrasing layer. Only an explicit False counts as a fault; None means the
# criterion could not be measured this rep and is not surfaced as a fault.
MAX_FAULTS_SURFACED = 2

# (verdict_key, exercise or None for any, fallback cue, description for the LLM)
_FAULTS = [
    ("trunk_ok",       None,    "keep your chest up",
     "Torso: leaned too far forward."),
    ("knee_travel_ok", "lunge", "keep your front knee over your ankle",
     "Front knee: drifted too far past the toes."),
    ("depth_ok",       "squat", "squat a little deeper",
     "Depth: did not reach parallel."),
    ("depth_ok",       "lunge", "drop your front knee lower",
     "Depth: front knee did not bend enough."),
]


def select_faults(verdict):
    """Return the prioritised, capped list of (cue, description) faults for a
    verdict. Deterministic: no language model involved."""
    exercise = verdict.get("exercise", "squat")
    faults = []
    for key, exo, cue, desc in _FAULTS:
        if exo is not None and exo != exercise:
            continue
        if verdict.get(key) is False:  # None (unmeasured) is not a fault
            faults.append((cue, desc))
    return faults[:MAX_FAULTS_SURFACED]


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
    """Translate a decided verdict into one coaching sentence.
    Falls back to a deterministic string if the model is unavailable."""
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
    """Deterministic coaching text, used when the model is unavailable. Covers
    exactly the same selected faults as the LLM path."""
    n = verdict["rep_number"]
    faults = select_faults(verdict)
    if not faults:
        return f"{_label(verdict)} {n}: great form, keep it up."
    return f"{_label(verdict)} {n}: " + " and ".join(cue for cue, _ in faults) + "."
