import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class TurnLog(BaseModel):
    turn_id: int
    agent_visible_message: str
    user_message: str
    internal_thoughts: str

class InterviewSession(BaseModel):
    participant_name: str
    turns: List[TurnLog] = Field(default_factory=list)
    final_feedback: Optional[str] = None

class SessionLogger:
    def __init__(self, filename: str = "interview_log.json"):
        self.filename = filename
        self.session: Optional[InterviewSession] = None
        self._start_new_session_if_needed()

    def _start_new_session_if_needed(self):
        # Always start fresh for a new run
        self.session = InterviewSession(
            participant_name="Candidate"
        )

    def start_session(self, participant_name: str):
        self.session.participant_name = participant_name
        self.save_log()

    def log_turn(self, turn_id: int, agent_msg: str, user_msg: str, thoughts: str):
        turn = TurnLog(
            turn_id=turn_id,
            agent_visible_message=agent_msg,
            user_message=user_msg,
            internal_thoughts=thoughts
        )
        self.session.turns.append(turn)
        self.save_log()

    def log_feedback(self, feedback: str):
        self.session.final_feedback = feedback
        self.save_log()

    def save_log(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write(self.session.model_dump_json(indent=2, exclude_none=True))
