import streamlit as st
from agents import build_graph

st.set_page_config(page_title="Deep Research AI", page_icon="🧠", layout="wide")

st.title("🔍 Deep Research AI")
st.markdown("A dual-agent system using Tavily, LangGraph & LangChain to research and generate answers.")

query = st.text_input("Enter your research question", placeholder="e.g. How is AI being used in climate change mitigation?")

if st.button("Run Research") and query:
    with st.spinner("Researching..."):
        graph_app = build_graph()
        result = graph_app.invoke({"query": query})

    st.subheader("📘 Final Answer")
    st.markdown(result["final_answer"])

    with st.expander("🗃️ Show Raw Research Context"):
        st.code(result["context"])

