import os
import io
import requests
import streamlit as st
from groq import Groq
from PIL import Image
from ddgs import DDGS

# =====================================================================
# 1. FRONTEND BRANDING OVERRIDES (CLAUDE-INSPIRED MINIMALIST THEME)
# =====================================================================
st.set_page_config(page_title="NexusAI OS", page_icon="🌐", layout="centered")

CLAUDE_THEME = """
<style>
    /* Main background - warm, clean minimalist vibe */
    .stApp {
        background-color: #f9f6f0 !important;
        color: #191919 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
    }
    
    /* Clean, understated header */
    h1 {
        color: #191919 !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px;
    }
    
    .stCaption {
        color: #6b6b6b !important;
    }
    
    /* Claude-style chat bubble layouts */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        padding: 1rem 0 !important;
        margin-bottom: 0px !important;
    }
    
    /* Assistant response background wrapper */
    div[data-testid="stChatMessageContent"] {
        color: #191919 !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }
    
    /* Refined, borderless look for input box */
    div[data-baseweb="input"] {
        background-color: #ffffff !important;
        border: 1px solid #e5e5e0 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    }
    
    input {
        color: #191919 !important;
    }
    
    /* Clean up helper elements */
    .stInfo, .stError {
        border-radius: 8px !important;
        background-color: #f0ede6 !important;
        color: #191919 !important;
        border: none !important;
    }
</style>
"""
st.markdown(CLAUDE_THEME, unsafe_allow_html=True)

st.title("🌐 NexusAI OS")
st.caption("A clean, minimalist platform with live web crawling and synthesis tools.")

# Read Groq API Key
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    st.error("🔑 Deployment Error: Missing GROQ_API_KEY in Streamlit Advanced Settings.")
    st.stop()

client = Groq(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================================================
# 2. CORE UTILITY AGENCIES (Web Crawling & Image Gen)
# =====================================================================

def tool_web_search(query: str) -> str:
    """Crawls live web search indices dynamically via DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            if not results:
                return "No matching live web data found."
            
            summary = ""
            for i, res in enumerate(results):
                summary += f"[{i+1}] Source: {res['title']}\nSnippet: {res['body']}\n\n"
            return summary
    except Exception as e:
        return f"Web crawler engine encountered an error: {str(e)}"


def tool_generate_image(prompt: str) -> Image.Image:
    """Generates visual assets dynamically using Hugging Face's serverless pipeline."""
    API_URL = "https://huggingface.co" 
    headers = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN', '')}"}
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=30)
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            return image
        return None
    except Exception:
        return None

# =====================================================================
# 3. INTERFACE RENDERER & ROUTING LOGIC
# =====================================================================

# Render conversational interface smoothly without box borders
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "image":
            st.image(msg["content"], caption=msg.get("caption"))
        else:
            st.markdown(msg["content"])

if user_input := st.chat_input("Ask NexusAI anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        
        if any(kw in user_input.lower() for kw in ["draw", "generate image", "create a picture of", "paint"]):
            st.info("🎨 Generating visual asset...")
            generated_img = tool_generate_image(user_input)
            
            if generated_img:
                st.image(generated_img, caption=f"Generated asset: '{user_input}'")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "type": "image", 
                    "content": generated_img, 
                    "caption": user_input
                })
            else:
                st.error("Failed to generate image. Please try another visual prompt.")
        
        else:
            context_data = ""
            if any(kw in user_input.lower() for kw in ["latest", "news", "current", "weather", "today", "search", "who is"]):
                st.info("🔍 Reviewing live web data...")
                context_data = tool_web_search(user_input)
            
            messages = [{
                "role": "system",
                "content": f"You are NexusAI, a thoughtful, helpful, and highly clear technical assistant. Format outputs beautifully using markdown. Avoid overly robotic phrases.\n\nLive Web Search Context:\n{context_data}"
            }]

            for msg in st.session_state.messages:
                if msg.get("type") != "image":
                    messages.append({
                        "role": "user" if msg["role"] == "user" else "assistant",
                        "content": msg["content"]
                    })

            try:
                response = client.chat.completions.create(
                    model='llama-3.3-70b-versatile',
                    messages=messages
                )
                
                ai_text = response.choices.message.content
                st.markdown(ai_text)
                st.session_state.messages.append({"role": "assistant", "content": ai_text})
                
            except Exception as e:
                st.error("⚠️ An unexpected server exception occurred.")
                with st.expander("Technical Trace Log"):
                    st.code(str(e))
