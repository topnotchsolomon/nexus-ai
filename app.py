import os
import io
import requests
import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
from duckduckgo_search import DDGS

# =====================================================================
# 1. UI BRANDING & SYSTEM INITIALIZATION
# =====================================================================
st.set_page_config(page_title="NexusAI OS", page_icon="🌐", layout="centered")
st.title("🌐 NexusAI OS")
st.caption("Custom Agent Platform with Live Web Crawling & Image Synthesis Tools")

# Initialize Gemini Client using your Environment Secret key
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("Deployment Error: Missing GEMINI_API_KEY in Environment Settings.")
    st.stop()

# Supports both legacy 'AIzaSy' and new 'AQ.Ab' credential formats
client = genai.Client(api_key=api_key)

# Persistent chat state management
if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================================================
# 2. PROPRIETARY UTILITY TOOLSETS
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
    # Free, open-source serverless community endpoint
    API_URL = "https://huggingface.co"
    
    try:
        response = requests.post(API_URL, json={"inputs": prompt}, timeout=30)
        image = Image.open(io.BytesIO(response.content))
        return image
    except Exception:
        return None

# =====================================================================
# 3. INTERFACE RENDERER & ROUTING LOGIC
# =====================================================================

# Draw historical logs on page refresh
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "image":
            st.image(msg["content"], caption=msg.get("caption"))
        else:
            st.markdown(msg["content"])

# Process active user interactions
if user_input := st.chat_input("Command NexusAI (e.g., 'draw a neon city' or 'latest tech news')..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        
        # Intent Check Type A: Image Synthesis requested
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
        
        # Intent Check Type B: Text Processing & Live Web Crawling
        else:
            context_data = ""
            # Automatically crawl the web if looking for fresh data
            if any(kw in user_input.lower() for kw in ["latest", "news", "current", "weather", "today", "search", "who is"]):
                st.info("🔍 Initializing Autonomous Web Crawler...")
                context_data = tool_web_search(user_input)
            
            # Formulate the custom model persona and inject live data context
            system_instruction = f"""
            You are NexusAI, an advanced autonomous OS engine. 
            Tone: High-intelligence, highly technical, clinical, authoritative.
            Behavior: Never say 'Sure, I can help!' or 'As an AI...'. State facts directly. 
            Always format your answers using clean markdown titles, sections, and structural lists.
            
            Live Real-Time Web Context:
            {context_data}
            """
            
            # FIXED: Format conversation using native dict structures to satisfy 'AQ.' keys
            contents = []
            for msg in st.session_state.messages:
                if msg.get("type") != "image":
                    contents.append({
                        "role": "user" if msg["role"] == "user" else "model",
                        "parts": [msg["content"]]
                    })

            try:
                # Query the model using your custom rule definitions
                response = client.models.generate_content(
                    model='gemini-1.5-flash',
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.3,
                    )
                )
                
                ai_text = response.text
                st.markdown(ai_text)
                st.session_state.messages.append({"role": "assistant", "content": ai_text})
                
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
