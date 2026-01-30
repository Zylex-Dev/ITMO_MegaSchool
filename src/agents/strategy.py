from typing import Dict, Any, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.state import InterviewState

class StrategyDecision(BaseModel):
    next_step: Literal["ask_question", "dig_deeper", "change_topic", "hint", "wrap_up"] = Field(description="The next move for the interviewer")
    topic: str = Field(description="The topic to focus on next")
    difficulty_change: int = Field(description="-1 (easier), 0 (same), 1 (harder)")
    directive: str = Field(description="Specific instructions for the Interviewer agent on what to ask/say")
    reasoning: str = Field(description="Why this decision was made")

class StrategyDirector:
    def __init__(self, model_name: str = "codestral-latest"):
        self.llm = ChatMistralAI(model=model_name, temperature=0.7)
        self.parser = JsonOutputParser(pydantic_object=StrategyDecision)
        
        self.system_prompt = """
        You are the Director of the Interview. You decide the flow.
        
        Inputs:
        - Technical Analysis: {tech_analysis}
        - Behavioral Analysis: {behavioral_analysis}
        - Current Topic: {current_topic}
        - Turns: {turn_count}
        
        Rules:
        1. If candidate is correct & confident -> Increase difficulty or Change Topic.
        2. If candidate is struggling -> Hint or Decrease difficulty.
        3. If Hallucination detected -> DIG DEEPER and excessive skepticism.
        4. If Off-topic -> Gently bring back to track.
        5. If turns > 10 -> WRAP UP.
        
        CRITICAL: Output ONLY valid JSON.
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Decide the next step.")
        ])
        
        self.chain = self.prompt | self.llm | self.parser

    def decide(self, state: InterviewState) -> Dict[str, Any]:
        tech = state.get("tech_analysis", {})
        behav = state.get("behavioral_analysis", {})
        
        try:
            result = self.chain.invoke({
                "tech_analysis": str(tech),
                "behavioral_analysis": str(behav),
                "current_topic": state.get("current_topic", "General"),
                "turn_count": state.get("turn_count", 0)
            })
            
            # Update state with the decision
            return {
                "strategy_directive": result["directive"],
                "current_topic": result["topic"],
                "difficulty_change": result["difficulty_change"] 
            }
        except Exception as e:
            # Return a valid default that forces progress rather than looping
            return {
                "strategy_directive": "Ask the next technical question based on candidate profile. Do not repeat introduction.", 
                "current_topic": "General Technical",
                "difficulty_change": 0,
                "error": str(e)
            }