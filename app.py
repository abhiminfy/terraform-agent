import os
import streamlit as st
import agent_input_parser  # your AI agent module
import io
import contextlib
import google.generativeai as genai  # For chatbot-style replies
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

# Streamlit page config
st.set_page_config(page_title="Terraform AI Generator", layout="centered")
st.title("🛠️ Terraform AI Generator + Chatbot")
st.caption("Describe your infrastructure needs. The AI will respond like a chatbot, generate Terraform code, estimate costs, and push to GitHub.")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User prompt input
user_prompt = st.chat_input("💬 Describe your infrastructure requirements...")

if user_prompt:
    st.session_state.chat_history.append(("user", user_prompt))

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Bot response using Gemini (chat-style)
    with st.chat_message("ai"):
        with st.spinner("💡 Thinking..."):
            try:
                gemini_reply = model.generate_content(user_prompt).text
            except Exception as e:
                gemini_reply = f"Error generating response: {str(e)}"
            st.markdown(gemini_reply)
            st.session_state.chat_history.append(("ai", gemini_reply))

    # Backend processing: Terraform code generation, cost estimation, GitHub push
    log_capture = io.StringIO()
    with contextlib.redirect_stdout(log_capture):
        try:
            terraform_code, cost_output, git_status = agent_input_parser.parse_user_input(user_prompt)
        except Exception as e:
            terraform_code = ""
            cost_output = ""
            git_status = ""
            st.error(f"❌ Error: {str(e)}")

    # Terraform code
    if terraform_code:
        with st.expander("📄 Terraform Configuration"):
            st.code(terraform_code, language="hcl")

    # Cost estimation
    if cost_output and isinstance(cost_output, str) and cost_output.strip():
        with st.expander("💰 Cost Estimate"):
            st.code(cost_output, language="bash")

    # GitHub push status
    if git_status:
        with st.expander("🔗 GitHub Push Status"):
            st.info(git_status)

    # Debug logs
    with st.expander("🐞 Debug Logs"):
        st.text(log_capture.getvalue())

# Show past chat
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)
