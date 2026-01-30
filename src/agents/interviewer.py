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
        Difficulty: {difficulty} (scale 1-5)
        Candidate Profile: {profile}
        
        Conversation History:
        {history}
        
        RULES:
        1. LANGUAGE: Always reply in the SAME language as the candidate's last message. If Russian, speak Russian.
        
        2. CANDIDATE QUESTIONS: If the candidate asks about the job, company, tasks, or trial period:
           - Answer their question professionally and helpfully
           - Then transition back to the interview with your next question
           - Example: "Отлично, на испытательном сроке вы будете работать с... А теперь вернемся к интервью: [question]"
        
        3. HALLUCINATION RESPONSE: If the directive mentions hallucination or false claims:
           - Politely but firmly correct the misinformation
           - Explain the correct facts
           - Ask a follow-up to verify understanding
        
        4. NO REPETITION: Check the history - do NOT ask about information already provided!
           - If candidate said they know Python, don't ask "what languages do you know?"
           - If they said their experience, don't ask "tell me about your experience"
        
        5. WRAP UP: If directive says "Wrap Up", thank them and end gracefully.
        
        6. NATURAL: Speak naturally, not robotically. Be professional but friendly.
        
        7. CONCISE: Keep questions short and focused.
        
        8. HIDDEN STATE: Never reveal internal state ("My strategy is...")
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Candidate's last message: {last_message}\n\nGenerate your response.")
        ])
        
        self.chain = self.prompt | self.llm

    def generate_response(self, state: InterviewState) -> Dict[str, Any]:
        directive = state.get("strategy_directive", "Ask a standard question.")
        topic = state.get("current_topic", "General")
        difficulty = state.get("difficulty_level", 1)
        
        messages = state.get("messages", [])
        last_message = messages[-1].content if messages else "Hello"
        
        # Get conversation history for context
        history_str = "\n".join([f"{m.type}: {m.content}" for m in messages[-8:]])
        
        # Get candidate profile
        profile = state.get("candidate_profile", {})
        profile_str = f"Name: {profile.get('name', 'Unknown')}, Position: {profile.get('position', 'N/A')}, Grade: {profile.get('grade', 'N/A')}, Skills: {profile.get('skills', [])}"
        
        try:
            response = self.chain.invoke({
                "directive": directive,
                "topic": topic,
                "difficulty": difficulty,
                "last_message": last_message,
                "history": history_str,
                "profile": profile_str
            })
            
            return {"messages": [response]}
        except Exception as e:
            fallback = "Could you please clarify?" 
            return {"messages": [AIMessage(content=fallback)]}
