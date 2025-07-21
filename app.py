# app.py
import os
import streamlit as st
import agent_input_parser  # your AI module

# Page setup
st.set_page_config(page_title="Terraform AI Generator", layout="centered")
st.title("🚀 Terraform AI Generator")
st.caption("Describe the infrastructure you need, get Terraform code, cost estimation, and GitHub auto-push.")

# User input
user_prompt = st.text_area(
    "Describe your infrastructure needs:",
    placeholder="e.g. Deploy a t2.micro EC2 instance in us-east-1",
    height=120
)

if st.button("Generate", type="primary"):
    if not user_prompt.strip():
        st.warning("⚠️ Please enter a valid prompt.")
        st.stop()

    import io
    import contextlib
    log_capture = io.StringIO()

    with st.spinner("⏳ Generating Terraform, estimating cost, and pushing to GitHub..."):
        with contextlib.redirect_stdout(log_capture):
            try:
                terraform_code, cost_estimate = agent_input_parser.parse_user_input(user_prompt)
                st.success("✅ Done! Your Terraform code is ready, estimated, and pushed to GitHub.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.stop()

    # Output
    st.subheader("📄 Terraform Configuration")
    st.code(terraform_code, language="hcl")

    st.subheader("💰 Cost Estimate")
    st.text(cost_estimate.strip())

    st.subheader("🔗 GitHub Push")
    st.info("The generated code was committed and pushed to your configured GitHub repo.")
