import os
import io
import requests
import streamlit as st
from groq import Groq  # 🛠️ Swapped from Google GenAI
from PIL import Image
from ddgs import DDGS

# =====================================================================
# 1. FRONTEND BRANDING OVERRIDES (RAW CSS INJECTION)
# =====================================================================
st.set_page_config(page_title="NexusAI OS", page_icon="🌐", layout="centered")

CYBERPUNK_THEME = """
<style>
    .stApp {
        background-color: #0d1117 !important;
        color: #58a6ff !important;
        font-family: 'SF Mono', Consolas, 'Courier New', monospace !important;
    }
    h1 {
        color: #ff79c6 !important;
        text-shadow: 0 0 12px rgba(255, 121, 198, 0.4);
        font-weight: 800 !important;
    }
    .stChatMessage {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        margin-bottom: 14px !important;
    }
    .stMarkdown {
        color: #c9d1d9 !important;
    }
    div[data-baseweb="input"] {
        background-color: #21262d !important;
        border: 1px solid #ff79c6 !important;
        border-radius: 4px !important;
    }
    input {
        color: #ff79c6 !important;
    }
</style>
"""
st.markdown(CYBERPUNK_THEME, unsafe_allow_html=True)

st.title("🌐 NexusAI OS")
st.caption("Custom Agent Platform with Live Web Crawling & Image Synthesis Tools")

# 🛠️ Read Groq API Key from Streamlit Secrets or environment variables
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    st.error("🔑 Deployment Error: Missing GROQ_API_KEY in Streamlit Advanced Settings.")
    st.stop()

# 🛠️ Initialize Groq Client
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
    # 📝 Note: Make sure to replace this URL or add your specific HF inference endpoint token if needed
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
            
            # 🛠️ System instruction formatted as a persistent system role message for Groq
            messages = [{
                "role": "system",
                "content": f"You are NexusAI, an advanced technical engine. Format outputs cleanly using markdown sections and lists.\n\nLive Web Search Context:\n{context_data}"
            }]

            # 🛠️ Format chat history conversion perfectly to match OpenAI/Groq standard chat structures
            for msg in st.session_state.messages:
                if msg.get("type") != "image":
                    messages.append({
                        "role": "user" if msg["role"] == "user" else "assistant",
                        "content": msg["content"]
                    })

            try:
                # 🛠️ Execution call via Groq API utilizing Llama-3.3-70b
                response = client.chat.completions.create(
                    model='llama-3.3-70b-versatile',
                    messages=messages
                )
                
                ai_text = response.choices[0].message.content
                st.markdown(ai_text)
                st.session_state.messages.append({"role": "assistant", "content": ai_text})
                
            except Exception as e:
                st.error("⚠️ Server Exception Encountered")
                with st.expander("Technical Trace Log"):
                    st.code(str(e))
