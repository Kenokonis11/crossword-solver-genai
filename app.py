import streamlit as st
import tempfile
import os
import PIL.Image
from llm_agent import CrosswordTutorAgent
from parsers.pdf_parser import PDFCrosswordParser

# --- Page Configuration ---
st.set_page_config(page_title="AI Crossword Tutor", page_icon="🧩")

# --- Session State Initialization ---
# Ensure puzzle_graph, agent, chat session, and message history exist in state.
if "puzzle_graph" not in st.session_state:
    st.session_state.puzzle_graph = None

if "agent" not in st.session_state:
    st.session_state.agent = CrosswordTutorAgent(
        api_key=os.environ.get("GOOGLE_API_KEY", "YOUR_API_KEY_HERE"),
        puzzle_graph=st.session_state.puzzle_graph,
    )

if "chat_session" not in st.session_state:
    st.session_state.chat_session = st.session_state.agent.start_chat()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Track which file has been parsed to avoid re-parsing on Streamlit reruns
if "parsed_file_name" not in st.session_state:
    st.session_state.parsed_file_name = None

# --- Sidebar UI ---
with st.sidebar:
    st.header("Puzzle Input Options")

    # --- File Upload: PDF, .puz, or image ---
    uploaded_file = st.file_uploader(
        "Upload a puzzle file",
        type=["pdf", "png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:
        # --- PDF Upload Handler ---
        if uploaded_file.name.endswith(".pdf"):
            if st.session_state.parsed_file_name != uploaded_file.name:
                with st.spinner("Parsing crossword PDF..."):
                    temp_path = None
                    try:
                        # Save uploaded bytes to a temporary file for PyMuPDF
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf"
                        ) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            temp_path = tmp.name

                        # Run the full parsing pipeline
                        parser = PDFCrosswordParser(source=temp_path)
                        graph = parser.build_graph()

                        # Store the graph and mark this file as parsed
                        st.session_state.puzzle_graph = graph
                        st.session_state.parsed_file_name = uploaded_file.name

                        # --- Autonomous Solver: Phase 1 + Phase 2 + Phase 3 ---
                        from solvers.autonomous_solver import AutonomousSolver
                        solver = AutonomousSolver(graph)
                        p1_locked = solver.execute_phase_1_pass()
                        p2_locked = solver.execute_phase_2_pass()
                        p3_locked = solver.execute_phase_3_pass()
                        solved_answers = solver.get_enriched_context()
                        st.session_state.solved_answers = solved_answers

                        # Re-initialize the agent with puzzle context + Ground Truth
                        st.session_state.agent = CrosswordTutorAgent(
                            api_key=os.environ.get("GOOGLE_API_KEY", "YOUR_API_KEY_HERE"),
                            puzzle_graph=graph,
                            solved_answers=solved_answers,
                        )
                        st.session_state.chat_session = (
                            st.session_state.agent.start_chat()
                        )
                        # Clear chat history so old context doesn't confuse the model
                        st.session_state.messages = []

                        summary = solver.get_solve_summary()
                        st.success(
                            f"✅ Loaded **{graph.title}** ({graph.width}x{graph.height}) "
                            f"— {len(graph.clues)} clues, {len(graph.intersections)} intersections\n\n"
                            f"🧠 Phase 1: **{p1_locked}** | "
                            f"Phase 2: **+{p2_locked}** | "
                            f"Phase 3: **+{p3_locked}** | "
                            f"Total: **{summary['solved_clues']}** / {summary['total_clues']} "
                            f"({summary['solve_percentage']}%)"
                        )

                    except ValueError as e:
                        st.error(
                            f"⚠️ Could not parse this PDF: {e}\n\n"
                            "We currently support vector-drawn crossword PDFs. "
                            "Rasterized/image-only PDFs will be supported in Phase 2."
                        )
                    except Exception as e:
                        st.error(f"Unexpected error parsing PDF: {e}")
                    finally:
                        # Clean up the temp file
                        if temp_path and os.path.exists(temp_path):
                            os.remove(temp_path)

        # --- Image Upload Handler ---
        elif uploaded_file.name.endswith((".png", ".jpg", ".jpeg")):
            image = PIL.Image.open(uploaded_file)
            st.session_state.puzzle_image = image
            st.success("Image loaded! Ask me to solve specific clues from the picture.")




    # --- Puzzle Status Indicator ---
    if st.session_state.puzzle_graph is not None:
        st.divider()
        g = st.session_state.puzzle_graph
        st.caption(f"📋 Active puzzle: **{g.title}** ({g.width}x{g.height})")

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
            try:
                if (
                    "puzzle_image" in st.session_state
                    and st.session_state.puzzle_image is not None
                ):
                    response = st.session_state.chat_session.send_message(
                        [prompt, st.session_state.puzzle_image]
                    )
                else:
                    response = st.session_state.chat_session.send_message(prompt)
                reply_text = response.text
            except ValueError as e:
                # Known GenAI SDK quirk: "Could not extract text from Part"
                # Happens when the model returns an empty or malformed response
                if "Part" in str(e):
                    reply_text = (
                        "I had a brief hiccup processing that — could you "
                        "rephrase or try again?"
                    )
                else:
                    raise
            except Exception as e:
                reply_text = f"Something went wrong: {e}. Please try again."

            st.markdown(reply_text)

    # 3. Append the model's textual response to state
    st.session_state.messages.append(
        {"role": "assistant", "content": reply_text}
    )
