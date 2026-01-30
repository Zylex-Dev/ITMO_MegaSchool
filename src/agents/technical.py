from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.state import InterviewState

# Define structured output for the evaluator
class TechEvaluation(BaseModel):
    is_correct: bool = Field(description="Is the candidate's answer technically correct?")
    confidence_score: float = Field(description="0.0 to 1.0 confidence in the answer correctness")
    hallucination_detected: bool = Field(description="Does the answer contain invented facts?")
    factual_errors: list[str] = Field(description="List of specific factual errors in the answer")
    missing_concepts: list[str] = Field(description="Key concepts that were missed")
    topics_covered: list[str] = Field(description="Technical topics discussed in this turn")
    reasoning: str = Field(description="Brief explanation of the evaluation")

class TechnicalEvaluator:
    def __init__(self, model_name: str = "codestral-latest"):
        self.llm = ChatMistralAI(model=model_name, temperature=0.0)
        self.parser = JsonOutputParser(pydantic_object=TechEvaluation)
        
        self.system_prompt = """
        You are a Technical Interview Evaluator. 
        Your job is to rigorously analyze the candidate's answer for technical accuracy.
        
        Current Topic: {topic}
        Difficulty Level: {difficulty}
        
        History:
        {history}
        
        Analyze the LAST user message.
        - Check for factual errors.
        - Check for "Hallucinations" (confident but wrong claims, e.g. "Python 4.0").
        - Identify missing key concepts.
        
        Output valid JSON matching the schema.
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{last_message}")
        ])
        
        self.chain = self.prompt | self.llm | self.parser

    def analyze(self, state: InterviewState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return {"tech_analysis": None}
            
        last_user_msg = messages[-1].content
        # In a real graph, we'd filter for the last HUMAN message
        
        # Format history briefly
        # Use 6 messages (3 full turns) for better context awareness
        history_str = "\n".join([f"{m.type}: {m.content}" for m in messages[-6:]])
        
        try:
            result = self.chain.invoke({
                "topic": state.get("current_topic", "General"),
                "difficulty": state.get("difficulty_level", 1),
                "history": history_str,
                "last_message": last_user_msg
            })
            return {"tech_analysis": result}
        except Exception as e:
            # Fallback for LLM errors
            return {"tech_analysis": {
                "error": str(e),
                "is_correct": False,
                "hallucination_detected": False,
                "reasoning": "Failed to analyze"
            }}
