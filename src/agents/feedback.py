from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.state import InterviewState

class SkillReview(BaseModel):
    skill_name: str
    status: str = Field(description="'Confirmed' or 'Gap'")
    comment: str

class FinalFeedback(BaseModel):
    grade: str = Field(description="Junior, Middle, Senior")
    hiring_recommendation: str = Field(description="No Hire, Hire, Strong Hire")
    confidence_score: float = Field(description="0-100%")
    technical_skills: List[SkillReview]
    soft_skills_summary: str
    roadmap: List[str] = Field(description="List of topics to improve")

class FeedbackGenerator:
    def __init__(self, model_name: str = "codestral-latest"):
        self.llm = ChatMistralAI(model=model_name, temperature=0.2)
        self.parser = JsonOutputParser(pydantic_object=FinalFeedback)
        
        self.system_prompt = """
        You are the Hiring Committee.
        Review the entire interview session and generate a structured report.
        
        Inputs:
        - Session History (Messages)
        - Candidate Name: {name}
        
        Outputs:
        1. Grade & Hiring Recommendation.
        2. Technical Analysis (Confirmed Skills vs Gaps).
        3. Soft Skills (Communication, Honesty).
        4. Learning Roadmap.
        
        IMPORTANT: Output ONLY raw JSON matching the schema. 
        Do NOT write "```json" or "```" or any markdown. Just the JSON object.
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Generate Report for:\n{history}")
        ])
        
        self.chain = self.prompt | self.llm | self.parser

    def generate(self, state: InterviewState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        history_str = "\n".join([f"{m.type}: {m.content}" for m in messages])
        
        try:
            result = self.chain.invoke({
                "name": "Candidate", # Could be parametrized
                "history": history_str
            })
            return {"feedback_report": result}
        except Exception as e:
            return {"feedback_report": {"error": str(e)}}