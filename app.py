import streamlit as st
import requests
import PIL.Image
from llm_agent import CrosswordTutorAgent
from parser import parse_puz_file
from csp_engine import CSPSolver
from orchestrator import run_solver_loop

# --- Page Configuration ---
st.set_page_config(page_title="AI Crossword Tutor", page_icon="🧩")

# --- Session State Initialization ---
# Ensure the agent, its active chat session, and the message history exist in state.
if "agent" not in st.session_state:
    st.session_state.agent = CrosswordTutorAgent(api_key="REDACTED_API_KEY")

if "chat_session" not in st.session_state:
    st.session_state.chat_session = st.session_state.agent.start_chat()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar UI ---
with st.sidebar:
    st.header("Puzzle Input Options")
    
    # Allow user to upload a local .puz file or an image
    uploaded_file = st.file_uploader("Upload a puzzle file", type=["puz", "png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".puz") and "solver" not in st.session_state:
            file_bytes = uploaded_file.getvalue()
            variables, intersections = parse_puz_file(file_bytes)
            st.session_state.solver = CSPSolver(variables, intersections)
            st.success("Puzzle Loaded!")
        elif uploaded_file.name.endswith((".png", ".jpg", ".jpeg")):
            image = PIL.Image.open(uploaded_file)
            st.session_state.puzzle_image = image
            st.success("Image loaded! Ask me to solve specific clues from the picture.")
        
    if st.button("Run Autonomous Solve"):
        if "solver" in st.session_state:
            with st.spinner("AI is solving the puzzle..."):
                run_solver_loop(st.session_state.solver)
            st.success("Solve Complete!")
        else:
            st.warning("Please upload a puzzle first.")
    
    st.divider()
    
    # Allow user to fetch a puzzle from the web (e.g., direct .puz URL)
    puzzle_source = st.text_input("Enter a direct .puz URL")
    if st.button("Fetch Puzzle"):
        if puzzle_source:
             try:
                 response = requests.get(puzzle_source)
                 if response.status_code == 200:
                     file_bytes = response.content
                     variables, intersections = parse_puz_file(file_bytes)
                     st.session_state.solver = CSPSolver(variables, intersections)
                     st.success("Successfully loaded puzzle from URL!")
                 else:
                     st.error(f"Failed to fetch puzzle. Status code: {response.status_code}")
             except requests.exceptions.RequestException as e:
                 st.error(f"Error fetching URL: {e}")
        else:
             st.warning("Please enter a url to fetch.")

# --- Main App & Chat Interface ---
st.title("🧩 AI Crossword Tutor")
st.markdown("Your interactive, neuro-symbolic crossword assistant. Ask me for a hint!")

# Re-render existing chat messages on each rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Chat Input Box
if prompt := st.chat_input("Ask for a hint or to solve a clue..."):
    
    # 1. Append user's message to state and print to screen
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
         st.markdown(prompt)

    # 2. Get AI Response via the active chat session (with a thinking spinner)
    with st.chat_message("assistant"):
         with st.spinner("Thinking..."):
              if "puzzle_image" in st.session_state and st.session_state.puzzle_image is not None:
                  response = st.session_state.chat_session.send_message([prompt, st.session_state.puzzle_image])
              else:
                  response = st.session_state.chat_session.send_message(prompt)
              st.markdown(response.text)
              
    # 3. Append the model's textual response to state
    st.session_state.messages.append({"role": "assistant", "content": response.text})
