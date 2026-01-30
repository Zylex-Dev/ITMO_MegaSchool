import sys
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from datetime import datetime

# Add the current directory to path so imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage, AIMessage
from src.graph import app
from src.logger import SessionLogger
from src.agents.feedback import FeedbackGenerator
from src.profile_parser import parse_candidate_intro, update_profile_from_message
from src.utils.formatter import beautify_log_file

# Stop commands detection
STOP_COMMANDS = ["exit", "quit", "stop", "стоп интервью", "стоп игра", "завершить", "давай фидбэк", "давай фидбек"]

def is_stop_command(user_input: str) -> bool:
    """Check if user input contains a stop command."""
    input_lower = user_input.lower()
    return any(cmd in input_lower for cmd in STOP_COMMANDS)

def format_internal_thoughts(tech_analysis: dict, behav_analysis: dict, strategy: str, reasoning: str) -> str:
    """Format internal agent thoughts for readable logging (Issue #6)."""
    
    # Extract tech analysis details
    tech_reasoning = tech_analysis.get('reasoning', 'N/A') if isinstance(tech_analysis, dict) else 'N/A'
    hallucination = tech_analysis.get('hallucination_detected', False) if isinstance(tech_analysis, dict) else False
    factual_errors = tech_analysis.get('factual_errors', []) if isinstance(tech_analysis, dict) else []
    missing = tech_analysis.get('missing_concepts', []) if isinstance(tech_analysis, dict) else []
    
    # Extract behavioral analysis details
    behav_observation = behav_analysis.get('observation', 'N/A') if isinstance(behav_analysis, dict) else 'N/A'
    honesty = behav_analysis.get('honesty_flag', 'N/A') if isinstance(behav_analysis, dict) else 'N/A'
    off_topic = behav_analysis.get('off_topic_attempt', False) if isinstance(behav_analysis, dict) else False
    candidate_question = behav_analysis.get('candidate_question', False) if isinstance(behav_analysis, dict) else False
    
    return f"""[Observer/Technical → Strategy Director]:
  Reasoning: {tech_reasoning}
  Hallucination Detected: {hallucination}
  Factual Errors: {factual_errors}
  Missing Concepts: {missing}

[Observer/Behavioral → Strategy Director]:
  Observation: {behav_observation}
  Honesty Assessment: {honesty}
  Off-topic Attempt: {off_topic}
  Candidate Asked Question: {candidate_question}

[Strategy Director → Interviewer]:
  Directive: {strategy}
  Reasoning: {reasoning}"""

def main():
    print(" initializing Multi-Agent Interview Coach...")
    
    # Initialize Logger
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(base_dir, 'interview_log.json')
    logger = SessionLogger(log_path)
    
    # Initialize Feedback Generator
    feedback_gen = FeedbackGenerator()
    
    participant_name = "Candidate"
    logger.start_session(participant_name)
    print(f" Session started for {participant_name}. Type 'стоп интервью' or 'exit' to finish and get feedback.")
    
    # Initial State
    state = {
        "messages": [],
        "candidate_profile": {},
        "interview_stage": "intro",
        "current_topic": "Introduction",
        "difficulty_level": 1,
        "turn_count": 0,
        "tech_analysis": None,
        "behavioral_analysis": None,
        "strategy_directive": None,
        "strategy_reasoning": None
    }
    
    turn_id = 1
    
    # Main Loop
    while True:
        try:
            print("\n" + "="*50)
            user_input = input(f"You ({participant_name}): ")
            
            # Check for stop commands - generate feedback and exit
            if is_stop_command(user_input):
                print("\n[System]: Generating final feedback report...")
                feedback_result = feedback_gen.generate(state)
                feedback_report = feedback_result.get("feedback_report", {})
                
                # Display feedback
                print("\n" + "="*50)
                print("📋 FINAL INTERVIEW FEEDBACK")
                print("="*50)
                print(json.dumps(feedback_report, indent=2, ensure_ascii=False))
                
                # Log feedback
                logger.log_feedback(json.dumps(feedback_report, indent=2, ensure_ascii=False))
                
                # Beautify log file
                beautify_log_file(log_path)
                
                print("\n[System]: Interview session completed. Thank you!")
                break
            
            # Add user message to state
            state["messages"].append(HumanMessage(content=user_input))
            state["turn_count"] = turn_id
            
            # Parse candidate profile from first few messages
            if turn_id <= 2:
                state["candidate_profile"] = update_profile_from_message(
                    state.get("candidate_profile", {}), 
                    user_input
                )
                # Update participant name if extracted
                if state["candidate_profile"].get("name"):
                    participant_name = state["candidate_profile"]["name"]
                    logger.session.participant_name = participant_name
            
            # Update interview stage based on turn count
            if turn_id == 1:
                state["interview_stage"] = "intro"
            elif turn_id <= 5:
                state["interview_stage"] = "main"
            elif turn_id <= 8:
                state["interview_stage"] = "behavioral"
            else:
                state["interview_stage"] = "closing"
            
            print(" [System]: Agents are thinking...")
            
            # Invoke the graph
            # The graph returns the final state after processing
            final_state = app.invoke(state)
            
            # Extract results
            agent_response = final_state["messages"][-1].content
            tech_analysis = final_state.get("tech_analysis", {})
            behav_analysis = final_state.get("behavioral_analysis", {})
            strategy = final_state.get("strategy_directive", "N/A")
            strategy_reasoning = final_state.get("strategy_reasoning", "N/A")
            
            # Display Agent Response
            print(f"\n[Interviewer]: {agent_response}")
            
            # Log with improved internal thoughts format (Issue #6)
            internal_thoughts = format_internal_thoughts(
                tech_analysis, behav_analysis, strategy, strategy_reasoning
            )
            logger.log_turn(turn_id, agent_response, user_input, internal_thoughts)
            
            # Update local state for next iteration
            state = final_state
            turn_id += 1
            
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    main()
