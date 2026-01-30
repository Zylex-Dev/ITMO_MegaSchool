from typing import Dict, Any, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.state import InterviewState

class StrategyDecision(BaseModel):
    next_step: Literal["ask_question", "dig_deeper", "change_topic", "hint", "wrap_up", "answer_candidate_question"] = Field(description="The next move for the interviewer")
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
        
        DECISION RULES (in priority order):
        
        1. CANDIDATE QUESTIONS: If behavioral_analysis shows candidate_question=True
           → Set next_step to "answer_candidate_question"
           → directive: "Answer the candidate's question about [topic], then ask a follow-up technical question"
           → This is NOT off-topic! Candidates asking about the job is NORMAL.
        
        2. HALLUCINATION DETECTED: If technical_analysis shows hallucination_detected=True
           → Set next_step to "dig_deeper"
           → directive: "Politely correct the false claim about [topic]. Explain the truth. Ask a follow-up to verify understanding."
           → difficulty_change: 0 (don't punish, just verify)
        
        3. CORRECT & CONFIDENT: If answer is correct and confidence is high
           → next_step: "ask_question" or "change_topic"
           → difficulty_change: +1 (increase challenge)
        
        4. STRUGGLING: If answer is wrong or candidate is uncertain
           → next_step: "hint" or "ask_question" with simpler version
           → difficulty_change: -1
        
        5. OFF-TOPIC (actual derailing like weather/politics):
           → next_step: "ask_question"  
           → directive: "Gently redirect back to the interview topic"
        
        6. WRAP UP: If turns > 10
           → next_step: "wrap_up"
           → directive: "Thank the candidate and conclude the interview"
        
        CRITICAL: Output ONLY valid JSON matching the schema.
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Decide the next step.")
        ])
        
        self.chain = self.prompt | self.llm | self.parser

    def decide(self, state: InterviewState) -> Dict[str, Any]:
        tech = state.get("tech_analysis", {})
        behav = state.get("behavioral_analysis", {})
        current_difficulty = state.get("difficulty_level", 1)
        
        try:
            result = self.chain.invoke({
                "tech_analysis": str(tech),
                "behavioral_analysis": str(behav),
                "current_topic": state.get("current_topic", "General"),
                "turn_count": state.get("turn_count", 0)
            })
            
            # Calculate new difficulty level (clamped to 1-5)
            difficulty_change = result.get("difficulty_change", 0)
            new_difficulty = max(1, min(5, current_difficulty + difficulty_change))
            
            # Update state with the decision including reasoning for logging
            return {
                "strategy_directive": result["directive"],
                "current_topic": result["topic"],
                "difficulty_level": new_difficulty,
                "strategy_reasoning": result.get("reasoning", "N/A")
            }
        except Exception as e:
            # Return a valid default that forces progress rather than looping
            return {
                "strategy_directive": "Ask the next technical question based on candidate profile. Do not repeat introduction.", 
                "current_topic": "General Technical",
                "difficulty_level": current_difficulty,  # Keep current level on error
                "strategy_reasoning": f"Error occurred: {str(e)}",
                "error": str(e)
            }