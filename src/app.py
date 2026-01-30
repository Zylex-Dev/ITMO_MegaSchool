import streamlit as st
import sys
import os
import json
from dotenv import load_dotenv
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage

# load env variables
load_dotenv()
# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph import app as graph_app
from src.logger import SessionLogger
from src.agents.feedback import FeedbackGenerator

# Page Config
st.set_page_config(page_title="AI Interview Coach", layout="wide")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "interview_state" not in st.session_state:
    st.session_state.interview_state = {
        "messages": [],
        "candidate_profile": {},
        "interview_stage": "intro",
        "current_topic": "Introduction",
        "turn_count": 0,
        "difficulty_level": 1,
        "tech_analysis": None,
        "behavioral_analysis": None,
        "strategy_directive": None
    }
if "turn_id" not in st.session_state:
    st.session_state.turn_id = 1
if "logger" not in st.session_state:
    # Initialize logger
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(base_dir, 'interview_log.json')
    st.session_state.logger = SessionLogger(log_path)
    st.session_state.logger.start_session("Streamlit User")
if "feedback_gen" not in st.session_state:
    st.session_state.feedback_gen = FeedbackGenerator()


# Sidebar - "The Brain"
with st.sidebar:
    st.header("Agent Thoughts")
    
    st.subheader("Strategy Directive")
    st.info(st.session_state.interview_state.get("strategy_directive", "Waiting to start..."))
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Topic")
        st.write(st.session_state.interview_state.get("current_topic", "N/A"))
    with col2:
        st.subheader("Difficulty")
        st.write(st.session_state.interview_state.get("difficulty_level", 1))

    with st.expander("Technical Analysis", expanded=True):
        st.json(st.session_state.interview_state.get("tech_analysis", {}))
        
    with st.expander("Behavioral Analysis", expanded=True):
        st.json(st.session_state.interview_state.get("behavioral_analysis", {}))

    st.markdown("---")
    if st.button("End Session & Generate Report", type="primary"):
        with st.spinner("Generating Final Feedback..."):
            report = st.session_state.feedback_gen.generate(st.session_state.interview_state)
            st.session_state.final_report = report.get("feedback_report")
            
            # Log it
            st.session_state.logger.log_feedback(json.dumps(st.session_state.final_report, indent=2))
        st.rerun()

# Display Final Report if it exists
if "final_report" in st.session_state and st.session_state.final_report:
    st.balloons()
    st.header("📋 Interview Feedback Report")
    
    rep = st.session_state.final_report
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Grade", rep.get("grade", "N/A"))
    col2.metric("Decision", rep.get("hiring_recommendation", "N/A"))
    col3.metric("Confidence", f"{rep.get('confidence_score', 0)*100:.0f}%")
    
    st.subheader("Technical Skills")
    for skill in rep.get("technical_skills", []):
        color = "green" if skill['status'] == 'Confirmed' else "red"
        st.markdown(f":{color}[{skill['skill_name']}]: {skill['comment']}")
        
    st.subheader("Soft Skills")
    st.write(rep.get("soft_skills_summary"))
    
    st.subheader("Roadmap")
    for item in rep.get("roadmap", []):
        st.markdown(f"- {item}")
        
    st.stop() # Stop rendering the chat input

# Main Chat Interface
st.title("👨‍💻 AI Technical Interviewer")

# Display Chat History
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Your answer..."):
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add to LangGraph State
    st.session_state.interview_state["messages"].append(HumanMessage(content=prompt))
    st.session_state.interview_state["turn_count"] = st.session_state.turn_id
    
    with st.spinner("Analyzing answer & Generating response..."):
        try:
            # Invoke Graph
            final_state = graph_app.invoke(st.session_state.interview_state)
            
            # Extract Response
            agent_msg = final_state["messages"][-1].content
            
            # Update Session State
            st.session_state.interview_state = final_state
            
            # Log
            internal_thoughts = f"Tech: {final_state.get('tech_analysis')}\nStrategy: {final_state.get('strategy_directive')}"
            st.session_state.logger.log_turn(
                st.session_state.turn_id, 
                agent_msg, 
                prompt, 
                internal_thoughts
            )
            st.session_state.turn_id += 1
            
            # Display Agent Response
            st.session_state.chat_history.append({"role": "assistant", "content": agent_msg})
            with st.chat_message("assistant"):
                st.markdown(agent_msg)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
