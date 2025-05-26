import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

for msg in st.session_state.conversation_history[1:]:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])
      
if prompt := st.chat_input("Ask a question:"):
    st.session_state.conversation_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
      
    with st.chat_message("assistant"):
        assistant_placeholder = st.empty()
        full_response = ""
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=st.session_state.conversation_history,
            stream=True
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                assistant_placeholder.markdown(full_response + "⬤")
        assistant_placeholder.markdown(full_response)

    st.session_state.conversation_history.append({"role": "assistant", "content": full_response})
