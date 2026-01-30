from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.state import InterviewState

class BehavioralEvaluation(BaseModel):
    clarity_score: int = Field(description="1-10 score on clarity of communication")
    confidence_score: int = Field(description="1-10 score on confidence")
    honesty_flag: str = Field(description="'honest', 'evasive', or 'deceptive'")
    engagement_level: str = Field(description="'high', 'medium', 'low'")
    off_topic_attempt: bool = Field(description="Is the candidate trying to change the subject?")
    observation: str = Field(description="Brief behavioral observation")

class BehavioralAnalyst:
    def __init__(self, model_name: str = "codestral-latest"):
        self.llm = ChatMistralAI(model=model_name, temperature=0.5)
        self.parser = JsonOutputParser(pydantic_object=BehavioralEvaluation)
        
        self.system_prompt = """
        You are a Behavioral Analyst for a technical interview.
        Focus on HOW the candidate answers, not just WHAT they say.
        
        - Assess confidence (Are they hedging? "Maybe", "I think").
        - Assess honesty (Are they admitting ignorance or bluffing?).
        - Check for "Off-topic" maneuvers (Trying to change subject to something irrelevant).
        
        History:
        {history}
        
        Analyze the LAST user message.
        Output structured JSON.
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{last_message}")
        ])
        
        self.chain = self.prompt | self.llm | self.parser

    def analyze(self, state: InterviewState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return {"behavioral_analysis": None}
            
        last_user_msg = messages[-1].content
        history_str = "\n".join([f"{m.type}: {m.content}" for m in messages[-3:]])
        
        try:
            result = self.chain.invoke({
                "history": history_str,
                "last_message": last_user_msg
            })
            return {"behavioral_analysis": result}
        except Exception as e:
            return {"behavioral_analysis": {"error": str(e), "observation": "Failed to analyze"}}
