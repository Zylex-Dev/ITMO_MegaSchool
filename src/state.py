from typing import List, Dict, Any, TypedDict, Annotated, Optional
import operator
from langchain_core.messages import BaseMessage

class InterviewState(TypedDict):
    """
    Global state for the interview process.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    candidate_profile: Dict[str, Any]  # Skills, confidence, gaps, name, position
    interview_stage: str # 'intro', 'main', 'code', 'behavioral', 'closing'
    current_topic: Optional[str]
    difficulty_level: int # 1-5
    turn_count: int
    
    # Analysis outputs
    tech_analysis: Optional[Dict[str, Any]]
    behavioral_analysis: Optional[Dict[str, Any]]
    strategy_directive: Optional[str]
    strategy_reasoning: Optional[str]  # Reasoning behind strategy decisions

