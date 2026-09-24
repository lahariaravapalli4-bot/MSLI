"""
nlp_engine.py: Maps detected ISL sign sequences into natural English sentences with Text-to-Speech.
"""

ISL_GRAMMAR_MAP = {
    # Greetings
    ("hello", "howareyou"): "Hello, how are you?",
    ("goodmorning", "teacher"): "Good morning, teacher.",
    ("teacher", "goodmorning"): "Good morning, teacher.",
    ("thankyou",): "Thank you very much.",
    
    # Campus / Academic
    ("i", "student"): "I am a student.",
    ("student", "school"): "The student goes to school.",
    ("teacher", "book"): "The teacher is reading a book.",
    
    # Emergency / Health
    ("i", "sick"): "I am not feeling well.",
    ("sick", "hospital"): "I am sick and need to go to the hospital.",
    ("doctor", "medicine"): "The doctor prescribed medicine.",
    ("hospital", "doctor"): "Please call a doctor at the hospital."
}


def synthesize_sentence(gloss_sequence: list[str]) -> str:
    if not gloss_sequence:
        return ""

    # Deduplicate consecutive identical detections (debouncing)
    cleaned = []
    for w in gloss_sequence:
        token = w.strip().lower()
        if not cleaned or cleaned[-1] != token:
            cleaned.append(token)

    # 1. Exact match lookup
    seq_tuple = tuple(cleaned)
    if seq_tuple in ISL_GRAMMAR_MAP:
        return ISL_GRAMMAR_MAP[seq_tuple]

    # 2. Subset keyword match
    token_set = set(cleaned)
    for rule_keys, sentence in ISL_GRAMMAR_MAP.items():
        if set(rule_keys).issubset(token_set):
            return sentence

    # 3. Fallback: Clean, capitalized string
    return " ".join(cleaned).capitalize() + "."


def speak_sentence(text: str):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 145)
        engine.say(text)
        engine.runAndWait()
    except Exception:
        print(f"[Speech Output]: {text}")


if __name__ == "__main__":
    test_run = ["goodmorning", "goodmorning", "teacher"]
    output = synthesize_sentence(test_run)
    print("Signs Detected   :", test_run)
    print("Synthesized Text :", output)
    speak_sentence(output)