from langgraph.graph import StateGraph, END

from src.state import InterviewState
from src.agents.technical import TechnicalEvaluator
from src.agents.behavioral import BehavioralAnalyst
from src.agents.strategy import StrategyDirector
from src.agents.interviewer import InterviewerAgent

tech_agent = TechnicalEvaluator()
behav_agent = BehavioralAnalyst()
strategy_agent = StrategyDirector()
interviewer_agent = InterviewerAgent()


def node_technical(state: InterviewState):
    return tech_agent.analyze(state)


def node_behavioral(state: InterviewState):
    return behav_agent.analyze(state)


def node_strategy(state: InterviewState):
    return strategy_agent.decide(state)


def node_interviewer(state: InterviewState):
    return interviewer_agent.generate_response(state)


workflow = StateGraph(InterviewState)

workflow.add_node("technical", node_technical)
workflow.add_node("behavioral", node_behavioral)
workflow.add_node("strategy", node_strategy)
workflow.add_node("interviewer", node_interviewer)

workflow.set_entry_point("technical")

workflow.add_edge("technical", "behavioral")
workflow.add_edge("behavioral", "strategy")
workflow.add_edge("strategy", "interviewer")
workflow.add_edge("interviewer", END)

app = workflow.compile()
