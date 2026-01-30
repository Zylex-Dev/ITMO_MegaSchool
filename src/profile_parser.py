"""
Candidate Profile Parser - extracts candidate information from first message.
Solves Issue #8: Add candidate_profile parsing from first message
"""
import re
from typing import Dict, Any, Optional


def parse_candidate_intro(message: str) -> Dict[str, Any]:
    """
    Extract candidate information from their introduction message.
    
    Expected patterns:
    - "Привет. Я Алекс, претендую на позицию Junior Backend Developer. Знаю Python, SQL и Git."
    - "Hi, I'm Alex, applying for Senior Frontend position. I know React, TypeScript."
    
    Returns:
        Dict with keys: name, position, grade, skills
    """
    profile = {
        "name": None,
        "position": None,
        "grade": None,
        "skills": [],
        "experience": None
    }
    
    # Extract name patterns
    name_patterns = [
        r"(?:Я|I'm|I am|Меня зовут|My name is)\s+([А-Яа-яЁёA-Za-z]+(?:\s+[А-Яа-яЁёA-Za-z]+)*)",
        r"^(?:Привет|Hi|Hello)[.\s!]*(?:Я|I'm|I am)?\s+([А-Яа-яЁёA-Za-z]+(?:\s+[А-Яа-яЁёA-Za-z]+)*)",
    ]
    for pattern in name_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            profile["name"] = match.group(1).strip()
            break
    
    # Extract grade
    grade_patterns = {
        "Junior": r"\b(junior|джун|джуниор|начинающ)\b",
        "Middle": r"\b(middle|мидл|миддл)\b", 
        "Senior": r"\b(senior|сеньор|синьор|старш)\b"
    }
    for grade, pattern in grade_patterns.items():
        if re.search(pattern, message, re.IGNORECASE):
            profile["grade"] = grade
            break
    
    # Extract position
    position_patterns = [
        r"(?:позиц\w*|position|role|вакансию?)\s+(?:на\s+)?([^.]+?)(?:\.|,|$)",
        r"(?:претенду\w*|applying for|apply\w* (?:for|as))\s+(?:на\s+)?([^.]+?)(?:\.|,|$)",
    ]
    for pattern in position_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            pos = match.group(1).strip()
            # Remove grade from position if present
            for grade in ["junior", "middle", "senior", "джун", "мидл", "сеньор"]:
                pos = re.sub(rf"\b{grade}\b\s*", "", pos, flags=re.IGNORECASE)
            profile["position"] = pos.strip()
            break
    
    # Extract skills
    skills_patterns = [
        r"(?:Знаю|I know|Skills?:?|Владею|Умею)\s+(.+?)(?:\.|$)",
        r"(?:опыт|experience)\s+(?:в|with|in)?\s*(.+?)(?:\.|$)",
    ]
    for pattern in skills_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            skills_str = match.group(1)
            # Split by common delimiters
            skills = re.split(r"[,;и&]|\band\b", skills_str)
            profile["skills"] = [s.strip() for s in skills if s.strip() and len(s.strip()) > 1]
            break
    
    # Extract experience duration
    exp_patterns = [
        r"(\d+)\s*(?:год|лет|years?|месяц|months?)",
        r"(?:опыт|experience)[:\s]+(\d+)",
    ]
    for pattern in exp_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            profile["experience"] = match.group(0)
            break
    
    return profile


def update_profile_from_message(current_profile: Dict[str, Any], message: str) -> Dict[str, Any]:
    """
    Update existing profile with new information from a message.
    Only updates empty fields.
    """
    new_info = parse_candidate_intro(message)
    
    for key, value in new_info.items():
        if value and not current_profile.get(key):
            current_profile[key] = value
    
    return current_profile
