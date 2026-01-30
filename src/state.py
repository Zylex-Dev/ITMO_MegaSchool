from typing import Any, TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage


class InterviewState(TypedDict):
    """
    Global state for the interview process.
    """

    messages: Annotated[list[BaseMessage], operator.add]
    candidate_profile: dict[str, Any]  # Skills, confidence, gaps, name, position
    interview_stage: str  # 'intro', 'main', 'code', 'behavioral', 'closing'
    current_topic: str | None
    difficulty_level: int  # 1-5
    turn_count: int

    tech_analysis: dict[str, Any] | None
    behavioral_analysis: dict[str, Any] | None
    strategy_directive: str | None
    strategy_reasoning: str | None  # Reasoning behind strategy decisions
