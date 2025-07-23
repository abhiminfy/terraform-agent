import os
import streamlit as st
import agent_input_parser  # your AI agent module
import io
import contextlib

# Streamlit page config
st.set_page_config(page_title="Terraform AI Generator", layout="centered")
st.title(" Terraform AI Generator")
st.caption("Describe your infrastructure needs, and this AI will generate Terraform code, estimate costs, and push to GitHub.")

# User prompt input
user_prompt = st.text_area(
    "🔍 Describe your infrastructure needs:",
    placeholder="e.g. Deploy a t2.micro EC2 instance in us-east-1",
    height=120
)

# Button trigger
if st.button(" Generate Infrastructure Plan"):
    if not user_prompt.strip():
        st.warning(" Please enter a prompt before generating.")
        st.stop()

    log_capture = io.StringIO()

    with st.spinner(" Generating Terraform code, estimating cost & pushing to GitHub..."):
        with contextlib.redirect_stdout(log_capture):
            try:
                terraform_code, cost_output, git_status = agent_input_parser.parse_user_input(user_prompt)
                st.success(" All tasks completed successfully!")
            except Exception as e:
                st.error(f" Error: {str(e)}")
                st.stop()

    # Terraform output
    st.subheader("📄 Terraform Configuration")
    if terraform_code:
        st.code(terraform_code, language="hcl")
    else:
        st.warning(" No Terraform code was generated.")

    # Cost estimation
    st.subheader(" Cost Estimate")
    if cost_output and isinstance(cost_output, str) and cost_output.strip():
        st.code(cost_output, language="bash")
    else:
        st.warning(" No cost estimate returned or estimation failed.")

    # GitHub push status
    st.subheader("🔗 GitHub Push Status")
    if git_status:
        st.info(git_status)
    else:
        st.warning(" GitHub push status not available.")

    # Debug log output
    with st.expander(" Debug Logs"):
        st.text(log_capture.getvalue())

