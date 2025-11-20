# streamlit_app.py
import streamlit as st
from typing import Generator
from groq import Groq

st.set_page_config(page_icon="💬", layout="wide", page_title="Groq Goes Brrrrrrrr...")

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

icon("🏎️")
st.subheader("Groq Chat Streamlit App", divider="rainbow", anchor=False)

# Create the client using secret (make sure you set GROQ_API_KEY in .streamlit/secrets.toml)
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    st.error("Groq API key not found in Streamlit secrets. Add GROQ_API_KEY to .streamlit/secrets.toml", icon="🚨")
    st.stop()

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# --- Only 3 functioning models (production/recommended ids) ---
# These are current Groq production model IDs (replace only if your console shows different available IDs)
models = {
    "llama-3.1-8b-instant": {"name": "LLaMA 3.1 8B (instant)", "tokens": 131072, "developer": "Meta"},
    "llama-3.3-70b-versatile": {"name": "LLaMA 3.3 70B (versatile)", "tokens": 128000, "developer": "Meta"}
    
}


# Layout: selector + tokens slider
col1, col2 = st.columns([3, 1])
with col1:
    model_option = st.selectbox(
        "Choose a model:",
        options=list(models.keys()),
        format_func=lambda x: models[x]["name"],
        index=0,  # default -> llama-3.1-8b-instant
    )

# clear history when model changes
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

max_tokens_range = models[model_option]["tokens"]
with col2:
    max_tokens = st.slider(
        "Max Tokens:",
        min_value=512,
        max_value=max_tokens_range,
        value=min(32768, max_tokens_range),
        step=512,
        help=f"Adjust the maximum number of tokens for model responses. Max for selected model: {max_tokens_range}"
    )

# Show chat history
for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "👨‍💻"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

def safe_extract_content(delta) -> str:
    """Extract 'content' from delta (handles dict-like or object-like chunk)."""
    if delta is None:
        return ""
    if isinstance(delta, dict):
        return delta.get("content", "") or ""
    return getattr(delta, "content", "") or ""

# Chat input: streaming handling
if prompt := st.chat_input("Enter your prompt here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👨‍💻"):
        st.markdown(prompt)

    full_response = ""  # always initialize so NameError cannot happen

    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            max_tokens=max_tokens,
            stream=True,
        )

        # Stream into UI progressively
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            for chunk in chat_completion:
                # robust extraction for different chunk shapes
                try:
                    choice = chunk.choices[0]
                except Exception:
                    try:
                        choice = chunk["choices"][0]
                    except Exception:
                        choice = None

                delta = None
                if choice is not None:
                    delta = getattr(choice, "delta", None) if not isinstance(choice, dict) else choice.get("delta")

                piece = safe_extract_content(delta)
                if piece:
                    full_response += piece
                    placeholder.markdown(full_response)

    except Exception as e:
        # Show the API error (e.g., model decommissioned) clearly to user
        st.error(f"API Error: {e}", icon="🚨")
        # Append the error as an assistant message so history shows it
        st.session_state.messages.append({"role": "assistant", "content": f"<API error: {e}>"})
    else:
        # If successful, append final response to history
        if isinstance(full_response, str) and full_response.strip() != "":
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "<no response received>"})
