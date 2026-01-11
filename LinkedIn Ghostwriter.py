import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import load_prompt

# 1. Page Configuration
st.set_page_config(page_title="LinkedIn Ghostwriter", page_icon="✍️")
st.title("✍️ LinkedIn Comment Ghostwriter")
st.markdown("Generate professional, high-engagement comments in seconds.")

# 2. Load Environment & Model
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("API Key missing! Please check your .env file.")
else:
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    
    # 3. Sidebar for Inputs
    with st.sidebar:
        st.header("Settings")
        persona = st.selectbox("Choose a Persona", 
                              ["Supportive Peer", "Thought Leader", "Skeptical Expert", "Funny/Witty"])
        goal = st.selectbox("Select Goal", 
                           ["Add Value", "Ask a Question", "Show Appreciation", "Challenge the Post"])

    # 4. Main Input Area
    post_content = st.text_area("Paste the LinkedIn Post here:", height=200, placeholder="What did the post say?")

    # 5. Execution
    if st.button("Generate Comment"):
        if post_content:
            with st.spinner("Drafting your comment..."):
                try:
                    # Load prompt from JSON
                    prompt_template = load_prompt("template.json")
                    chain = prompt_template | model
                    
                    # Run Chain
                    response = chain.invoke({
                        "post_content": post_content,
                        "persona": persona,
                        "goal": goal
                    })
                    
                    # 6. Display Result (Ready for Copy-Paste)
                    st.success("Done! You can copy the comment below:")
                    
                    # Using st.code gives the user a built-in copy button!
                    st.code(response.content, language=None)
                    
                    st.info("💡 Tip: Use the 'Copy' button in the top right of the box above.")
                    
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
        else:
            st.warning("Please paste some content first!")