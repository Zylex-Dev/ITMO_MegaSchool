from typing import Dict, Any, List, Literal, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.state import InterviewState


class SkillDetail(BaseModel):
    """Technical skill that was confirmed during interview."""
    skill_name: str = Field(description="Name of the technical skill")
    evidence: str = Field(description="Quote or description of what the candidate demonstrated")


class GapDetail(BaseModel):
    """Knowledge gap identified during interview."""
    topic: str = Field(description="Topic where the candidate showed weakness")
    candidate_response: str = Field(description="What the candidate said (quote if applicable)")
    correct_answer: str = Field(description="The correct answer or explanation they should have provided")


class SoftSkillsAnalysis(BaseModel):
    """Analysis of candidate's soft skills."""
    clarity: int = Field(ge=1, le=10, description="1-10 score on clarity of communication")
    honesty: Literal["Honest", "Evasive", "Deceptive"] = Field(description="Honesty assessment")
    engagement: Literal["High", "Medium", "Low"] = Field(description="Level of engagement in conversation")
    summary: str = Field(description="Brief summary of soft skills observations")


class RoadmapItem(BaseModel):
    """Learning recommendation for the candidate."""
    topic: str = Field(description="Topic to study")
    priority: Literal["High", "Medium", "Low"] = Field(description="Priority level")
    resources: Optional[List[str]] = Field(default=None, description="Optional links to documentation or articles")


class FinalFeedback(BaseModel):
    """Final interview feedback report - matches technical specification."""
    grade: Literal["Junior", "Middle", "Senior"] = Field(description="Candidate level based on answers")
    hiring_recommendation: Literal["No Hire", "Hire", "Strong Hire"] = Field(description="Hiring decision")
    confidence_score: int = Field(ge=0, le=100, description="0-100% confidence in the assessment")
    
    # Technical Analysis
    confirmed_skills: List[SkillDetail] = Field(description="Skills that were verified as present")
    knowledge_gaps: List[GapDetail] = Field(description="Topics where candidate showed weaknesses WITH correct answers")
    
    # Soft Skills
    soft_skills: SoftSkillsAnalysis = Field(description="Analysis of communication, honesty, engagement")
    
    # Roadmap
    roadmap: List[RoadmapItem] = Field(description="Personalized learning plan with priorities")


class FeedbackGenerator:
    def __init__(self, model_name: str = "codestral-latest"):
        self.llm = ChatMistralAI(model=model_name, temperature=0.2)
        self.parser = JsonOutputParser(pydantic_object=FinalFeedback)
        
        self.system_prompt = """
        You are the Hiring Committee reviewing a technical interview.
        Generate a comprehensive, structured feedback report.
        
        Candidate Name: {name}
        
        OUTPUT MUST BE VALID JSON with EXACTLY these field names (lowercase with underscores):
        
        {{
          "grade": "Junior" | "Middle" | "Senior",
          "hiring_recommendation": "No Hire" | "Hire" | "Strong Hire",
          "confidence_score": 0-100,
          "confirmed_skills": [
            {{"skill_name": "...", "evidence": "..."}}
          ],
          "knowledge_gaps": [
            {{"topic": "...", "candidate_response": "...", "correct_answer": "..."}}
          ],
          "soft_skills": {{
            "clarity": 1-10,
            "honesty": "Honest" | "Evasive" | "Deceptive",
            "engagement": "High" | "Medium" | "Low",
            "summary": "..."
          }},
          "roadmap": [
            {{"topic": "...", "priority": "High" | "Medium" | "Low", "resources": ["url1", "url2"]}}
          ]
        }}
        
        EVALUATION GUIDELINES:
        
        1. GRADE: 
           - Junior: Basic understanding, needs guidance
           - Middle: Solid fundamentals, can work independently  
           - Senior: Deep expertise, can lead and mentor
        
        2. KNOWLEDGE GAPS: For EACH wrong answer, include:
           - What the candidate said (quote them)
           - The CORRECT answer they should have known
        
        3. SOFT SKILLS: 
           - Clarity: How well they explain concepts (1-10)
           - Honesty: Did they admit when they didn't know?
           - Engagement: Were they interested and asking good questions?
        
        4. ROADMAP: Prioritized list of topics to study with resource links
        
        CRITICAL:
        - Use EXACTLY the field names shown above (lowercase with underscores)
        - Do NOT include fictional technologies (e.g., "Python 4.0") in recommendations
        - Output ONLY the JSON object, no markdown or extra text
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Interview History:\n{history}\n\nGenerate the feedback report as JSON.")
        ])
        
        self.chain = self.prompt | self.llm | self.parser

    def _normalize_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize LLM response to ensure consistent field names."""
        if not result:
            return result
            
        normalized = {}
        
        # Grade
        normalized["grade"] = (
            result.get("grade") or result.get("GRADE") or result.get("Grade") or "Junior"
        )
        
        # Hiring recommendation
        normalized["hiring_recommendation"] = (
            result.get("hiring_recommendation") or 
            result.get("HIRING RECOMMENDATION") or 
            result.get("Hiring Recommendation") or
            result.get("hiring_rec") or
            "No Hire"
        )
        
        # Confidence score
        conf = result.get("confidence_score") or result.get("CONFIDENCE SCORE") or result.get("Confidence Score") or 0
        normalized["confidence_score"] = conf if isinstance(conf, (int, float)) else 0
        
        # Confirmed skills
        normalized["confirmed_skills"] = (
            result.get("confirmed_skills") or 
            result.get("CONFIRMED SKILLS") or 
            result.get("Confirmed Skills") or
            result.get("technical_skills") or
            []
        )
        
        # Knowledge gaps
        normalized["knowledge_gaps"] = (
            result.get("knowledge_gaps") or 
            result.get("KNOWLEDGE GAPS") or 
            result.get("Knowledge Gaps") or
            []
        )
        
        # Soft skills
        soft = result.get("soft_skills") or result.get("SOFT SKILLS") or result.get("Soft Skills") or {}
        if isinstance(soft, dict):
            normalized["soft_skills"] = {
                "clarity": soft.get("clarity") or soft.get("Clarity") or 5,
                "honesty": soft.get("honesty") or soft.get("Honesty") or "Honest",
                "engagement": soft.get("engagement") or soft.get("Engagement") or "Medium",
                "summary": soft.get("summary") or soft.get("Summary") or ""
            }
        else:
            normalized["soft_skills"] = {"clarity": 5, "honesty": "Honest", "engagement": "Medium", "summary": str(soft)}
        
        # Roadmap
        normalized["roadmap"] = (
            result.get("roadmap") or 
            result.get("ROADMAP") or 
            result.get("Roadmap") or
            []
        )
        
        return normalized

    def generate(self, state: InterviewState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        candidate_profile = state.get("candidate_profile", {})
        candidate_name = candidate_profile.get("name", "Кандидат")
        
        # Check if there is enough history to generate feedback
        # We need at least some user interaction (e.g. 2 user messages)
        user_messages = [m for m in messages if m.type == 'human']
        if len(user_messages) < 1:
            return {"feedback_report": {
                "grade": "N/A",
                "hiring_recommendation": "No Hire",
                "confidence_score": 0,
                "confirmed_skills": [],
                "knowledge_gaps": [],
                "soft_skills": {
                    "clarity": 0,
                    "honesty": "N/A",
                    "engagement": "N/A",
                    "summary": "Интервью было завершено до начала содержательной беседы. Недостаточно данных для оценки."
                },
                "roadmap": []
            }}

        history_str = "\n".join([f"{m.type}: {m.content}" for m in messages])
        
        try:
            result = self.chain.invoke({
                "name": candidate_name,
                "history": history_str
            })
            # Normalize the response to handle different naming conventions
            normalized = self._normalize_response(result)
            return {"feedback_report": normalized}
        except Exception as e:
            # Return a minimal valid response on error
            return {"feedback_report": {
                "error": str(e),
                "grade": "Junior",
                "hiring_recommendation": "No Hire",
                "confidence_score": 0,
                "confirmed_skills": [],
                "knowledge_gaps": [],
                "soft_skills": {
                    "clarity": 5,
                    "honesty": "Honest",
                    "engagement": "Medium",
                    "summary": "Не удалось оценить из-за ошибки."
                },
                "roadmap": []
            }}