import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from datetime import datetime

# Add the current directory to path so imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage, AIMessage
from src.graph import app
from src.logger import SessionLogger

def main():
    print(" initializing Multi-Agent Interview Coach...")
    
    # Initialize Logger
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(base_dir, 'interview_log.json')
    logger = SessionLogger(log_path)
    
    participant_name = "Candidate"
    logger.start_session(participant_name)
    print(f" Session started for {participant_name}. Type 'exit' to quit.")
    
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
        "strategy_directive": None
    }
    
    turn_id = 1
    
    # Main Loop
    while True:
        try:
            print("\n" + "="*50)
            user_input = input(f"You ({participant_name}): ")
            if user_input.lower() in ["exit", "quit", "stop"]:
                logger.log_feedback("User terminated session manually.")
                break
            
            # Add user message to state
            state["messages"].append(HumanMessage(content=user_input))
            state["turn_count"] = turn_id
            
            print(" [System]: Agents are thinking...")
            
            # Invoke the graph
            # The graph returns the final state after processing
            final_state = app.invoke(state)
            
            # Extract results
            agent_response = final_state["messages"][-1].content
            tech_analysis = final_state.get("tech_analysis", {})
            behav_analysis = final_state.get("behavioral_analysis", {})
            strategy = final_state.get("strategy_directive", "N/A")
            
            # Display Agent Response
            print(f"\n[Interviewer]: {agent_response}")
            
            # Log the turn clearly
            internal_thoughts = f"[Technical]: {tech_analysis}\n[Behavioral]: {behav_analysis}\n[Strategy]: {strategy}"
            logger.log_turn(turn_id, agent_response, user_input, internal_thoughts)
            
            # Update local state for next iteration
            state = final_state
            turn_id += 1
            
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    main()
