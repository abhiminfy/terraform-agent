import os 
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Safe import for agent logic
try:
    import agent_input_parser
except ImportError:
    st.error("❌ Failed to import agent_input_parser.py. Make sure it exists and contains `process_user_prompt()`.")
    st.stop()

# Streamlit page config
st.set_page_config(page_title="Terraform AI Generator + Chatbot", layout="centered")
st.title("Terraform AI ")
st.caption("Talk freely. Ask general questions or describe your infra needs. The AI will respond smartly and generate Terraform scripts only when needed.")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_prompt = st.chat_input("💬 Ask anything or describe your infrastructure...")

# Handle new input
if user_prompt:
    # Append and show user message
    st.session_state.chat_history.append(("user", user_prompt))

    # Get response from agent
    try:
        if hasattr(agent_input_parser, "process_user_prompt"):
            result = agent_input_parser.process_user_prompt(user_prompt)
        else:
            raise AttributeError("`process_user_prompt()` not found in agent_input_parser.py")

        response_type = result.get("type")

        if response_type == "terraform":
            ai_message = "✅ Infrastructure prompt detected. Here's your Terraform configuration and deployment info."
            terraform_code = result.get("terraform_code", "")
            cost_estimate = result.get("cost_estimate", "")
            github_status = result.get("github_status", "")

            ai_detailed_msg = ai_message

            if terraform_code:
                ai_detailed_msg += "\n\n**📄 Terraform Configuration:**\n```hcl\n" + terraform_code + "\n```"
            if cost_estimate:
                ai_detailed_msg += "\n\n**💰 Cost Estimate:**\n```bash\n" + cost_estimate + "\n```"
            if github_status:
                ai_detailed_msg += "\n\n**🔗 GitHub Push Status:**\n" + github_status

            st.session_state.chat_history.append(("ai", ai_detailed_msg))

        elif response_type == "clarify":
            clarification = result.get("content", "")
            ai_message = "🤔 I need a bit more detail before generating infrastructure.\n" + clarification
            st.session_state.chat_history.append(("ai", ai_message))

        elif response_type == "chat":
            content = result.get("content", "I'm here to help!")
            st.session_state.chat_history.append(("ai", content))

        elif response_type == "error":
            error_msg = f"❌ Error: {result.get('error', 'Unknown error')}"
            st.session_state.chat_history.append(("ai", error_msg))

        else:
            st.session_state.chat_history.append(("ai", "🤖 Received an unrecognized response type."))

    except Exception as e:
        error_msg = f"❌ Unexpected error: {str(e)}"
        st.session_state.chat_history.append(("ai", error_msg))

# Display chat history (older at top, newer at bottom)
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)


