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
st.title("🛠️ Terraform AI Generator + Chatbot")
st.caption("Talk freely. Ask general questions or describe your infra needs. The AI will respond smartly and generate Terraform scripts only when needed.")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_prompt = st.chat_input("💬 Ask anything or describe your infrastructure...")

# Handle input
if user_prompt:
    st.session_state.chat_history.append(("user", user_prompt))

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("ai"):
        with st.spinner("💡 Thinking..."):
            try:
                # Process the user prompt
                if hasattr(agent_input_parser, "process_user_prompt"):
                    result = agent_input_parser.process_user_prompt(user_prompt)
                else:
                    raise AttributeError("`process_user_prompt()` not found in agent_input_parser.py")

                response_type = result.get("type")

                if response_type == "terraform":
                    st.markdown("✅ Infrastructure prompt detected. Here's the response:")

                    if result.get("terraform_code"):
                        with st.expander("📄 Terraform Configuration"):
                            st.code(result["terraform_code"], language="hcl")

                    if result.get("cost_estimate"):
                        with st.expander("💰 Cost Estimate"):
                            st.code(result["cost_estimate"], language="bash")

                    if result.get("github_status"):
                        with st.expander("🔗 GitHub Push Status"):
                            st.info(result["github_status"])

                    ai_message = "Here's your Terraform configuration and deployment info. Let me know if you'd like to modify or expand it."
                    st.markdown(ai_message)
                    st.session_state.chat_history.append(("ai", ai_message))

                elif response_type == "clarify":
                    ai_message = "🤔 I need a bit more detail before generating infrastructure. Can you clarify?"
                    st.markdown(ai_message)
                    st.markdown(result.get("content", ""))
                    st.session_state.chat_history.append(("ai", ai_message + "\n" + result.get("content", "")))

                elif response_type == "chat":
                    content = result.get("content", "I'm here to help!")
                    st.markdown(content)
                    st.session_state.chat_history.append(("ai", content))

                elif response_type == "error":
                    error_msg = f"❌ Error: {result.get('error', 'Unknown error')}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(("ai", error_msg))

                else:
                    st.warning("🤖 Received an unrecognized response type.")
                    st.session_state.chat_history.append(("ai", "🤖 Received an unrecognized response type."))

            except Exception as e:
                error_msg = f"❌ Unexpected error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append(("ai", error_msg))

# Display chat history
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)



