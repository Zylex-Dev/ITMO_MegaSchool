import json
import os
import re


def clean_thoughts(thoughts: str) -> str:
    """
    Cleans the internal thoughts by removing newlines and extra spaces,
    but keeping the full original content.
    """
    if not thoughts or not isinstance(thoughts, str):
        return thoughts

    cleaned = thoughts.replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def beautify_log_file(filepath: str):
    """
    Reads the log file, beautifies final_feedback and internal_thoughts,
    and saves the result to a new file with 'beautiful_' prefix.
    """
    if not os.path.exists(filepath):
        return

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        feedback_raw = data.get("final_feedback")
        if feedback_raw and isinstance(feedback_raw, str):
            try:
                feedback_dict = json.loads(feedback_raw)
                data["final_feedback"] = feedback_dict
            except json.JSONDecodeError:
                pass

        turns = data.get("turns", [])
        for turn in turns:
            thoughts = turn.get("internal_thoughts")
            if thoughts:
                turn["internal_thoughts"] = clean_thoughts(thoughts)

        dirname = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        beautiful_filename = f"beautiful_{filename}"
        new_path = os.path.join(dirname, beautiful_filename)

        with open(new_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Beautified log saved to: {new_path}")

    except Exception as e:
        print(f"Error beautifying log file: {e}")
