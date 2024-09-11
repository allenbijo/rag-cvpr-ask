import streamlit as st
import pickle
import fetch_papers, vectorsaving, language_model
from huggingface_hub import login
import os
login(os.getenv('hf_token'))

st.title("Arxiv Paper RAG")

with st.sidebar:
    st.markdown('# Papers')

    if st.button("Fetch Papers"):
        with st.spinner("Fetching papers..."):
            paper_names = fetch_papers.get_paper_names()
            papers = fetch_papers.fetch_arxiv_papers(paper_names)
            vectorsaving.vectorize_papers(papers, paper_names)

    with open("papers.pkl", "rb") as f:
        try:
            existing_data = pickle.load(f)
        except:
            existing_data = []
    st.markdown(f'Fetched {len(existing_data)} papers')
    st.markdown('\n\n'.join(existing_data))

if "chain" not in st.session_state:
    st.session_state['model'] = language_model.prepare_model()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        print(type(chain:=language_model.setup(st.session_state['model'])))
    response = chain.invoke(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
