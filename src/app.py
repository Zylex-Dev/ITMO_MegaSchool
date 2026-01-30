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
from src.profile_parser import update_profile_from_message

# Page Config
st.set_page_config(page_title="AI Интервьюер", layout="wide")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "interview_state" not in st.session_state:
    st.session_state.interview_state = {
        "messages": [],
        "candidate_profile": {},
        "interview_stage": "intro",
        "current_topic": "Знакомство",
        "turn_count": 0,
        "difficulty_level": 1,
        "tech_analysis": {},
        "behavioral_analysis": {},
        "strategy_directive": "Ожидание представления кандидата...",
        "strategy_reasoning": None
    }
if "turn_id" not in st.session_state:
    st.session_state.turn_id = 1
if "logger" not in st.session_state:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(base_dir, 'interview_log.json')
    st.session_state.logger = SessionLogger(log_path)
    st.session_state.logger.start_session("Кандидат")
if "feedback_gen" not in st.session_state:
    st.session_state.feedback_gen = FeedbackGenerator()


def normalize_feedback(report):
    """Нормализация отчета с учетом разных вариантов именования полей от LLM."""
    if not report or not isinstance(report, dict):
        return None
    
    normalized = {}
    
    # Грейд
    normalized["grade"] = (
        report.get("grade") or 
        report.get("GRADE") or 
        report.get("Grade") or 
        "N/A"
    )
    
    # Рекомендация по найму    
    normalized["hiring_recommendation"] = (
        report.get("hiring_recommendation") or 
        report.get("HIRING RECOMMENDATION") or 
        report.get("Hiring Recommendation") or
        report.get("hiring_rec") or
        "N/A"
    )
    
    # Уверенность в оценке
    confidence = (
        report.get("confidence_score") or 
        report.get("CONFIDENCE SCORE") or 
        report.get("Confidence Score") or
        report.get("confidence") or
        0
    )
    if isinstance(confidence, (int, float)):
        normalized["confidence_score"] = confidence if confidence > 1 else confidence * 100
    else:
        normalized["confidence_score"] = 0
    
    # Подтвержденные навыки
    normalized["confirmed_skills"] = (
        report.get("confirmed_skills") or 
        report.get("CONFIRMED SKILLS") or 
        report.get("Confirmed Skills") or
        report.get("technical_skills") or
        []
    )
    
    # Пробелы в знаниях
    normalized["knowledge_gaps"] = (
        report.get("knowledge_gaps") or 
        report.get("KNOWLEDGE GAPS") or 
        report.get("Knowledge Gaps") or
        []
    )
    
    # Софт-скиллы
    soft = (
        report.get("soft_skills") or 
        report.get("SOFT SKILLS") or 
        report.get("Soft Skills") or
        report.get("soft_skills_summary") or
        {}
    )
    normalized["soft_skills"] = soft
    
    # Roadmap
    normalized["roadmap"] = (
        report.get("roadmap") or 
        report.get("ROADMAP") or 
        report.get("Roadmap") or
        []
    )
    
    return normalized


# Боковая панель - "Мозг агента"
with st.sidebar:
    st.header("Мысли агента")
    
    st.subheader("Директива стратегии")
    directive = st.session_state.interview_state.get("strategy_directive")
    st.info(directive if directive else "Ожидание начала...")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Тема")
        st.write(st.session_state.interview_state.get("current_topic", "Н/Д"))
    with col2:
        st.subheader("Сложность")
        st.write(f"Уровень {st.session_state.interview_state.get('difficulty_level', 1)}/5")

    with st.expander("Технический анализ", expanded=True):
        tech = st.session_state.interview_state.get("tech_analysis")
        if tech and isinstance(tech, dict) and len(tech) > 0:
            st.json(tech)
        else:
            st.caption("_Ожидание первого ответа..._")
        
    with st.expander("Поведенческий анализ", expanded=True):
        behav = st.session_state.interview_state.get("behavioral_analysis")
        if behav and isinstance(behav, dict) and len(behav) > 0:
            st.json(behav)
        else:
            st.caption("_Ожидание первого ответа..._")

    # Профиль кандидата если доступен
    profile = st.session_state.interview_state.get("candidate_profile", {})
    if profile and any(profile.values()):
        with st.expander("Профиль кандидата", expanded=False):
            st.json(profile)

    st.markdown("---")
    if st.button("🏁 Завершить и получить отчёт", type="primary"):
        with st.spinner("Генерация итогового отчёта..."):
            report = st.session_state.feedback_gen.generate(st.session_state.interview_state)
            st.session_state.final_report = report.get("feedback_report")
            
            if st.session_state.final_report:
                st.session_state.logger.log_feedback(json.dumps(st.session_state.final_report, indent=2, ensure_ascii=False))
        st.rerun()

# Отображение итогового отчета
if "final_report" in st.session_state and st.session_state.final_report:
    st.balloons()
    st.header("Итоговый отчёт по интервью")
    
    rep = normalize_feedback(st.session_state.final_report)
    
    if rep:
        col1, col2, col3 = st.columns(3)
        
        # Перевод грейда
        grade = rep.get("grade", "N/A")
        # grade_ru = {"Junior": "Джуниор", "Middle": "Мидл", "Senior": "Сеньор"}.get(grade, grade)
        col1.metric("Грейд", grade)
        
        # Перевод рекомендации
        rec = rep.get("hiring_recommendation", "N/A")
        rec_ru = {"No Hire": "Не нанимать", "Hire": "Нанять", "Strong Hire": "Точно нанять"}.get(rec, rec)
        col2.metric("Решение", rec_ru)
        
        col3.metric("Уверенность", f"{rep.get('confidence_score', 0):.0f}%")
        
        # Подтвержденные навыки
        st.subheader("✅ Подтверждённые навыки")
        confirmed = rep.get("confirmed_skills", [])
        if confirmed:
            for skill in confirmed:
                if isinstance(skill, dict):
                    skill_name = skill.get("skill_name") or skill.get("topic") or skill.get("Topic") or str(skill)
                    evidence = skill.get("evidence") or skill.get("comment") or ""
                    st.markdown(f"- **{skill_name}**: {evidence}" if evidence else f"- **{skill_name}**")
                else:
                    st.markdown(f"- {skill}")
        else:
            st.caption("_Подтверждённые навыки не зафиксированы_")
        
        # Пробелы в знаниях
        st.subheader("❌ Пробелы в знаниях")
        gaps = rep.get("knowledge_gaps", [])
        if gaps:
            for gap in gaps:
                if isinstance(gap, dict):
                    topic = gap.get("topic") or gap.get("Topic") or "Неизвестно"
                    response = gap.get("candidate_response") or gap.get("Candidate Statement") or gap.get("candidate_statement") or ""
                    correct = gap.get("correct_answer") or gap.get("Correct Answer") or gap.get("correctAnswer") or ""
                    
                    st.markdown(f"**{topic}**")
                    if response:
                        st.markdown(f"> _Кандидат сказал:_ {response}")
                    if correct:
                        st.success(f"✓ Правильный ответ: {correct}")
                else:
                    st.markdown(f"- {gap}")
        else:
            st.caption("_Пробелов в знаниях не выявлено_")
            
        # Софт-скиллы
        st.subheader("💬 Soft skills")
        soft = rep.get("soft_skills", {})
        if soft:
            if isinstance(soft, dict):
                clarity = soft.get("clarity") or soft.get("Clarity") or "Н/Д"
                honesty = soft.get("honesty") or soft.get("Honesty") or "Н/Д"
                engagement = soft.get("engagement") or soft.get("Engagement") or "Н/Д"
                summary = soft.get("summary") or soft.get("Summary") or ""
                
                # Перевод значений
                honesty_ru = {"Honest": "Честный", "Evasive": "Уклончивый", "Deceptive": "Обманчивый"}.get(honesty, honesty)
                engagement_ru = {"High": "Высокая", "Medium": "Средняя", "Low": "Низкая"}.get(engagement, engagement)
                
                cols = st.columns(3)
                cols[0].metric("Ясность", f"{clarity}/10" if isinstance(clarity, int) else clarity)
                cols[1].metric("Честность", honesty_ru)
                cols[2].metric("Вовлечённость", engagement_ru)
                if summary:
                    st.write(summary)
            elif isinstance(soft, str):
                st.write(soft)
        else:
            st.caption("_Анализ мягких навыков недоступен_")
        
        # Roadmap
        st.subheader("📚 План обучения")
        roadmap = rep.get("roadmap", [])
        if roadmap:
            for item in roadmap:
                if isinstance(item, dict):
                    topic = item.get("topic") or item.get("Topic") or "Тема"
                    priority = item.get("priority") or item.get("Priority") or ""
                    resources = item.get("resources") or item.get("Resources") or []
                    
                    priority_ru = {"High": "Высокий", "Medium": "Средний", "Low": "Низкий"}.get(priority, priority)
                    # priority_emoji = {"High": "1", "Высокий": "1", "Medium": "2", "Средний": "2", "Low": "3", "Низкий": "3"}.get(priority, "")
                    
                    st.markdown(f"**{topic}** _{priority_ru}_" if priority else f"**{topic}**")
                    
                    if resources:
                        for res in resources:
                            st.markdown(f"  - [{res}]({res})" if res.startswith("http") else f"  - {res}")
                else:
                    st.markdown(f"- {item}")
        else:
            st.caption("_Рекомендации отсутствуют_")
    else:
        st.error("Ошибка при обработке отчёта")
        st.json(st.session_state.final_report)
        
    st.stop()

# Основной интерфейс чата
st.title("👨‍💻 AI Технический интервьюер")
st.caption("Представьтесь и расскажите о своих навыках. Для завершения введите 'стоп интервью'")

# История чата
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ввод пользователя
if prompt := st.chat_input("Ваш ответ..."):
    # Определение стоп-команд
    STOP_COMMANDS = ["exit", "quit", "stop", "стоп интервью", "стоп игра", "завершить", "давай фидбэк", "давай фидбек", "завершить интервью", "закончить"]
    is_stop = any(cmd in prompt.lower() for cmd in STOP_COMMANDS)
    
    # Добавление сообщения в историю
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Если стоп-команда - генерируем отчёт
    if is_stop:
        with st.spinner("Генерация итогового отчёта..."):
            report = st.session_state.feedback_gen.generate(st.session_state.interview_state)
            st.session_state.final_report = report.get("feedback_report")
            
            if st.session_state.final_report:
                st.session_state.logger.log_feedback(json.dumps(st.session_state.final_report, indent=2, ensure_ascii=False))
        st.rerun()
    else:
        # Добавляем в состояние LangGraph
        st.session_state.interview_state["messages"].append(HumanMessage(content=prompt))
        st.session_state.interview_state["turn_count"] = st.session_state.turn_id
        
        # Парсинг профиля кандидата из первых сообщений
        if st.session_state.turn_id <= 2:
            st.session_state.interview_state["candidate_profile"] = update_profile_from_message(
                st.session_state.interview_state.get("candidate_profile", {}),
                prompt
            )
            profile = st.session_state.interview_state["candidate_profile"]
            if profile.get("name"):
                st.session_state.logger.session.participant_name = profile["name"]
                st.session_state.logger.save_log()
        
        # Обновление этапа интервью
        turn = st.session_state.turn_id
        if turn == 1:
            st.session_state.interview_state["interview_stage"] = "intro"
        elif turn <= 5:
            st.session_state.interview_state["interview_stage"] = "main"
        elif turn <= 8:
            st.session_state.interview_state["interview_stage"] = "behavioral"
        else:
            st.session_state.interview_state["interview_stage"] = "closing"
        
        with st.spinner("Анализ ответа и генерация вопроса..."):
            try:
                # Вызов графа агентов
                final_state = graph_app.invoke(st.session_state.interview_state)
                
                # Извлечение ответа
                agent_msg = final_state["messages"][-1].content
                
                # Обновление состояния сессии
                st.session_state.interview_state = final_state
                
                # Логирование
                tech_analysis = final_state.get('tech_analysis') or {}
                behav_analysis = final_state.get('behavioral_analysis') or {}
                strategy = final_state.get('strategy_directive', 'Н/Д')
                
                internal_thoughts = f"""[Наблюдатель/Технический]: {tech_analysis.get('reasoning', 'Н/Д') if isinstance(tech_analysis, dict) else 'Н/Д'}
  - Галлюцинация: {tech_analysis.get('hallucination_detected', False) if isinstance(tech_analysis, dict) else False}
  - Пропущенные концепции: {tech_analysis.get('missing_concepts', []) if isinstance(tech_analysis, dict) else []}

[Наблюдатель/Поведенческий]: {behav_analysis.get('observation', 'Н/Д') if isinstance(behav_analysis, dict) else 'Н/Д'}
  - Честность: {behav_analysis.get('honesty_flag', 'Н/Д') if isinstance(behav_analysis, dict) else 'Н/Д'}
  - Оффтопик: {behav_analysis.get('off_topic_attempt', False) if isinstance(behav_analysis, dict) else False}

[Стратег → Интервьюер]: {strategy}"""
                
                st.session_state.logger.log_turn(
                    st.session_state.turn_id, 
                    agent_msg, 
                    prompt, 
                    internal_thoughts
                )
                st.session_state.turn_id += 1
                
                # Отображение ответа агента
                st.session_state.chat_history.append({"role": "assistant", "content": agent_msg})
                with st.chat_message("assistant"):
                    st.markdown(agent_msg)
                    
            except Exception as e:
                st.error(f"Ошибка: {str(e)}")
