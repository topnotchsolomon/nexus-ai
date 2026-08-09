import os
import io
import requests
import streamlit as st
from google import genai
from PIL import Image
from duckduckgo_search import DDGS

# =====================================================================
# 1. PREMIUM CUSTOM FRONTEND INJECTION (HTML & CSS)
# =====================================================================
st.set_page_config(page_title="NexusAI OS", page_icon="🌐", layout="centered")

# Custom CSS Stylesheet applied globally across the user interface
CUSTOM_CSS = """
<style>
    /* Main container styling */
    .stApp {
        background-color: #0b0f19 !important;
        color: #00ffcc !important;
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    /* Top title adjustments */
    h1 {
        color: #ff007f !important;
        text-shadow: 0 0 10px #ff007f, 0 0 20px #ff007f;
        font-size: 2.8rem !important;
        font-weight: bold;
    }
    
    /* Chat message area custom styling */
    .stChatMessage {
        background-color: #121824 !important;
        border: 1px solid #00ffcc !important;
        border-radius: 8px !important;
        box-shadow: 0 0 8px rgba(0, 255, 204, 0.2);
        margin-bottom: 12px !important;
        padding: 15px !important;
    }
    
    /* Input box styling customization */
    div[data-baseweb="input"] {
        background-color: #1a2333 !important;
        border: 1px solid #ff007f !important;
        border-radius: 4px !important;
    }
    
    input {
        color: #00ffcc !important;
    }
</style>
"""
# Render the HTML/CSS template to the user's viewport
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("🌐 NexusAI OS")
st.caption("Custom Agent Platform with Live Web Crawling & Image Synthesis Tools")

# Initialize Gemini Client securely using Environment Secrets
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("🔑 Deployment Error: Missing GEMINI_API_KEY in Streamlit Advanced Settings.")
    st.stop()

client = genai.Client(api_key=api_key)

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
    try:
        response = requests.post(API_URL, json={"inputs": prompt}, timeout=30)
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            return image
        return None
    except Exception:
        return None

# =====================================================================
# 3. INTERFACE RENDERER & ROUTING LOGIC
# =====================================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "image":
            st.image(msg["content"], caption=msg.get("caption"))
        else:
            st.markdown(msg["content"])

if user_input := st.chat_input("Command NexusAI (e.g., 'draw a neon city' or 'latest tech news')..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        
        if any(kw in user_input.lower() for kw in ["draw", "generate image", "create a picture of", "paint"]):
            st.info("🎨 Initializing Serverless Image Generation Engines...")
            generated_img = tool_generate_image(user_input)
            
            if generated_img:
                st.image(generated_img, caption=f"Synthesized by NexusAI Engine: '{user_input}'")
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
                st.info("🔍 Initializing Autonomous Web Crawler...")
                context_data = tool_web_search(user_input)
            
            system_instruction = f"""You are NexusAI, an advanced autonomous OS engine.
Tone: High-intelligence, clinical, authoritative.
Behavior: State facts directly without boilerplate conversational fluff or generic introductory statements.
Format using clean markdown layout patterns.

Live Real-Time Web Context:
{context_data}
"""

            contents = []
            for msg in st.session_state.messages:
                if msg.get("type") != "image":
                    contents.append({
                        "role": "user" if msg["role"] == "user" else "model",
                        "parts": [{"text": msg["content"]}]
                    })

            try:
                # FIXED: Removed 'temperature' argument to comply with strict AQ authentication endpoints
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=contents,
                    config={"system_instruction": system_instruction}
                )
                
                ai_text = response.text
                st.markdown(ai_text)
                st.session_state.messages.append({"role": "assistant", "content": ai_text})
                
            except Exception as e:
                st.error("⚠️ Server Exception Encountered")
                with st.expander("Technical Trace Log"):
                    st.code(str(e))
