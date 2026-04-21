"""
capstone_streamlit.py — ShopEasy FAQ Bot UI
Run with: streamlit run capstone_streamlit.py
"""

import streamlit as st
import uuid
from agent import build_graph, ask, KNOWLEDGE_BASE

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopEasy Support Bot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Cache expensive resources (LLM, embedder, ChromaDB, compiled graph)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_agent():
    """Load and compile the LangGraph agent once."""
    app = build_graph()
    return app

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛒 ShopEasy Support Bot")
    st.markdown("**Agentic AI Capstone 2026**")
    st.markdown("---")

    st.markdown("### About This Bot")
    st.markdown(
        "This AI assistant helps ShopEasy customers get instant answers "
        "about orders, shipping, returns, payments, and more — available 24/7."
    )

    st.markdown("### Topics I Can Help With")
    topics = [doc["topic"] for doc in KNOWLEDGE_BASE]
    for topic in topics:
        st.markdown(f"• {topic}")

    st.markdown("---")
    st.markdown(f"**Session ID:** `{st.session_state.thread_id[:8]}...`")

    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.user_name = ""
        st.rerun()

    st.markdown("---")
    st.markdown("**Contact ShopEasy:**")
    st.markdown("📧 support@shopeasy.com")
    st.markdown("📞 1800-123-4567 (Toll-free)")
    st.markdown("💬 WhatsApp: +91-9000-123456")

# ─────────────────────────────────────────────────────────────────────────────
# Main chat interface
# ─────────────────────────────────────────────────────────────────────────────
st.title("🛒 ShopEasy Customer Support")
st.caption("Ask me anything about orders, shipping, returns, payments, and more!")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            with st.expander("🔍 Details", expanded=False):
                meta = msg["meta"]
                col1, col2 = st.columns(2)
                col1.metric("Route", meta.get("route", "N/A"))
                col2.metric("Faithfulness", f"{meta.get('faithfulness', 0):.2f}")
                if meta.get("sources"):
                    st.markdown(f"**Sources:** {', '.join(meta['sources'])}")

# Welcome message
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 Hello! I'm ShopEasy's AI support assistant. I can help you with:\n\n"
            "- 📦 Order tracking and status\n"
            "- 🔄 Returns and refunds\n"
            "- 🚚 Shipping and delivery\n"
            "- 💳 Payments and EMI options\n"
            "- ❌ Cancellations\n"
            "- 🎟️ Discount codes\n"
            "- And much more!\n\n"
            "Feel free to tell me your name so I can personalise my responses. How can I help you today?"
        )

# Chat input
if prompt := st.chat_input("Type your question here..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                app = load_agent()
                result = ask(app, prompt, st.session_state.thread_id)

                answer = result.get("answer", "I'm sorry, I couldn't generate a response. Please try again.")
                route = result.get("route", "N/A")
                faithfulness = result.get("faithfulness", 0.0)
                sources = result.get("sources", [])
                user_name = result.get("user_name", "")

                if user_name:
                    st.session_state.user_name = user_name

                st.markdown(answer)

                with st.expander("🔍 Response Details", expanded=False):
                    col1, col2 = st.columns(2)
                    col1.metric("Route", route.capitalize())
                    col2.metric("Faithfulness Score", f"{faithfulness:.2f}")
                    if sources:
                        st.markdown(f"**Knowledge Sources:** {', '.join(sources)}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "meta": {
                        "route": route,
                        "faithfulness": faithfulness,
                        "sources": sources,
                    }
                })

            except Exception as e:
                error_msg = f"⚠️ An error occurred: {str(e)}. Please check your GROQ_API_KEY and try again."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
