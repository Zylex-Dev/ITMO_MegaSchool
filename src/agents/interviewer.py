from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage

from src.state import InterviewState

class InterviewerAgent:
    def __init__(self, model_name: str = "codestral-latest"):
        self.llm = ChatMistralAI(model=model_name, temperature=0.7)
        
        self.system_prompt = """
        You are a Professional Technical Interviewer.
        Your goal is to assess the candidate effectively while maintaining a professional environment.
        
        Current Directive from Strategy Director:
        "{directive}"
        
        Topic: {topic}
        Difficulty: {difficulty}
        
        Rules:
        1. LANGUAGE: Always reply in the SAME language as the candidate's last message. If they speak Russian, speak Russian. If English, speak English.
        2. Speak naturally. Do not sound robotic.
        3. Follow the Directive strictly (e.g. if told to "dig deeper", ask a follow-up specific to the previous answer).
        4. If the directive says "Wrap Up", thank them and end the interview.
        5. Do NOT reveal the internal state (don't say "My strategy is to...").
        6. Keep questions concise.
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Candidate's last message: {last_message}\n\nGenerate the response.")
        ])
        
        self.chain = self.prompt | self.llm

    def generate_response(self, state: InterviewState) -> Dict[str, Any]:
        directive = state.get("strategy_directive", "Ask a standard question.")
        topic = state.get("current_topic", "General")
        difficulty = state.get("difficulty_level", 1)
        
        messages = state.get("messages", [])
        last_message = messages[-1].content if messages else "Hello"
        
        try:
            response = self.chain.invoke({
                "directive": directive,
                "topic": topic,
                "difficulty": difficulty,
                "last_message": last_message
            })
            
            return {"messages": [response]}
        except Exception as e:
            fallback = "Could you please clarify?" 
            # Simple multilingual fallback attempt? better to keep english default if error
            return {"messages": [AIMessage(content=fallback)]}