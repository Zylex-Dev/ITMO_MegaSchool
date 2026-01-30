from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from src.state import InterviewState
from src.agents.technical import TechnicalEvaluator
from src.agents.behavioral import BehavioralAnalyst
from src.agents.strategy import StrategyDirector
from src.agents.interviewer import InterviewerAgent
from src.logger import SessionLogger

# Initialize Agents
tech_agent = TechnicalEvaluator()
behav_agent = BehavioralAnalyst()
strategy_agent = StrategyDirector()
interviewer_agent = InterviewerAgent()

# Node Functions
def node_technical(state: InterviewState):
    return tech_agent.analyze(state)

def node_behavioral(state: InterviewState):
    return behav_agent.analyze(state)

def node_strategy(state: InterviewState):
    return strategy_agent.decide(state)

def node_interviewer(state: InterviewState):
    return interviewer_agent.generate_response(state)

# Graph Construction
workflow = StateGraph(InterviewState)

workflow.add_node("technical", node_technical)
workflow.add_node("behavioral", node_behavioral)
workflow.add_node("strategy", node_strategy)
workflow.add_node("interviewer", node_interviewer)

# Define edges
# Parallel execution for evaluators
workflow.set_entry_point("technical") 
workflow.add_edge("technical", "behavioral") # Sequential for simplicity in this version, or use parallel branching
# Ideally: Start -> [Tech, Behav] -> Strategy. 
# In LangGraph, we can do Start -> Tech, Start -> Behav. Then both -> Strategy.
# But we need a synchronization point. Strategy needs both.
# Let's keep it sequential for safety: Tech -> Behavioral -> Strategy -> Interviewer.

workflow.add_edge("technical", "behavioral")
workflow.add_edge("behavioral", "strategy")
workflow.add_edge("strategy", "interviewer")
workflow.add_edge("interviewer", END)

# Compile
app = workflow.compile()
