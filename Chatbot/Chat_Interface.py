import streamlit as st
from openai import OpenAI

class ChatInterface:
    def __init__(self):
        self.client = OpenAI()
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize Streamlit session state."""
        st.session_state.setdefault("pdf_data", None)
        st.session_state.setdefault("vectorstore", None)
        st.session_state.setdefault("conversation_history", [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on tables and text extracted from a PDF."}
        ])

    def display_conversation(self):
        """Display conversation history."""
        for msg in st.session_state.conversation_history[1:]:
            with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                st.markdown(msg["content"])

    def process_user_input(self, prompt, vector_store_manager):
        """Process user input and generate response."""
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if vector_store_manager.vectorstore:
            with st.chat_message("assistant"):
                try:
                    docs = vector_store_manager.search(prompt)
                    combined_context = "\n\n".join([doc.page_content for doc in docs])
                    response_container = st.empty()
                    full_response = ""
                    response_stream = self.client.chat.completions.create(
                        model="gpt-4.1-nano",
                        messages=[
                            *st.session_state.conversation_history,
                            {"role": "system", "content": f"Relevant content:\n{combined_context}"},
                            {"role": "user", "content": prompt},
                        ],
                        stream=True
                    )
                    for chunk in response_stream:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            full_response += delta
                            response_container.markdown(full_response + "⬤ ")
                    response_container.markdown(full_response)
                    st.session_state.conversation_history.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
        else:
            st.error("Please upload and process a PDF first.")