from PDF_processor import PDFProcessor
from VectorStore import VectorStoreManager
from Chat_Interface import ChatInterface
from dotenv import load_dotenv
import streamlit as st

def main():
    load_dotenv()
    pdf_processor = PDFProcessor()
    vector_store_manager = VectorStoreManager()
    chat_interface = ChatInterface()

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        try:
            with st.spinner("Processing PDF..."):
                st.session_state.pdf_data, pdf_text = pdf_processor.extract_tables_and_text(uploaded_file)
                st.session_state.vectorstore = vector_store_manager.create_vector_store(st.session_state.pdf_data, pdf_text)
                st.success("PDF processed successfully! You can now ask questions.")
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")

    chat_interface.display_conversation()
    
    if prompt := st.chat_input("Ask a question about the PDF"):
        chat_interface.process_user_input(prompt, vector_store_manager)

if __name__ == "__main__":
    main()