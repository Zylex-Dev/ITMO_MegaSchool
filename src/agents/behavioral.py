from typing import Any
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
    off_topic_attempt: bool = Field(
        description="Is the candidate trying to derail the interview with irrelevant topics?"
    )
    candidate_question: bool = Field(
        description="Is the candidate asking a legitimate question about the job/company?"
    )
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
        - Check for "Off-topic" attempts.

        IMPORTANT DISTINCTION for off_topic_attempt:
        - Off-topic = TRUE only for irrelevant topics like weather, politics, personal life, jokes, etc.
        - Off-topic = FALSE for legitimate candidate questions about:
          * The job role, responsibilities, tasks
          * The company, team, projects
          * Technical stack, methodologies used
          * Onboarding, trial period expectations

        These are NORMAL parts of an interview! Set candidate_question=true when the candidate asks about the job.

        History:
        {history}

        Analyze the LAST user message.
        Output structured JSON.
        """

        self.prompt = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("human", "{last_message}")]
        )

        self.chain = self.prompt | self.llm | self.parser

    def analyze(self, state: InterviewState) -> dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return {"behavioral_analysis": None}

        last_user_msg = messages[-1].content
        # Use 6 messages (3 full turns) for better context awareness
        history_str = "\n".join([f"{m.type}: {m.content}" for m in messages[-6:]])

        try:
            result = self.chain.invoke(
                {"history": history_str, "last_message": last_user_msg}
            )
            return {"behavioral_analysis": result}
        except Exception as e:
            return {
                "behavioral_analysis": {
                    "error": str(e),
                    "observation": "Failed to analyze",
                }
            }
