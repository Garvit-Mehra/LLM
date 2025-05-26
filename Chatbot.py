import streamlit as st
from openai import OpenAI
import dotenv
import camelot
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import PyPDF2
import os
import pandas as pd

dotenv.load_dotenv()
client = OpenAI()

st.session_state.setdefault("pdf_data", None)
st.session_state.setdefault("vectorstore", None)
st.session_state.setdefault("conversation_history", [
    {"role": "system", "content": "You are a helpful assistant that answers questions based on tables and text extracted from a PDF."}
])

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def clean_multiline_rows(df):
    cleaned_rows = []
    current_row = []
    for _, row in df.iterrows():
        if row[0] and not current_row:
            if current_row:
                cleaned_rows.append(current_row)
            current_row = row.tolist()
        else:
            for i, cell in enumerate(row):
                if cell:
                    current_row[i] = f"{current_row[i] or ''} {cell}".strip()
    if current_row:
        cleaned_rows.append(current_row)
    return pd.DataFrame(cleaned_rows, columns=df.columns)

def extract_tables_and_text_from_pdf(file):
    file.seek(0)
    pdf_text = extract_text_from_pdf(file)
    
    temp_file_path = "temp_pdf.pdf"
    file.seek(0)
    with open(temp_file_path, "wb") as f:
        f.write(file.read())
    
    try:
        tables = camelot.read_pdf(
            temp_file_path,
            pages="all",
            flavor="lattice",
            split_text=True,
            strip_text='\n',
        )
        
        if not tables:
            st.warning("No tables detected in lattice mode. Trying stream mode...")
            tables = camelot.read_pdf(
                temp_file_path,
                pages="all",
                flavor="stream",
                row_tol=15,
                columns_separator=' '
            )
        
        if not tables:
            st.error("No tables detected in either mode. Consider using LLMWhisperer for complex PDFs.")
            return [], pdf_text
        
        table_data = []
        for table in tables:
            cleaned_df = clean_multiline_rows(table.df)
            table_markdown = cleaned_df.to_markdown(index=False)
            combined_content = f"Table (Page {table.page}):\n{table_markdown}\n\nPDF Text:\n{pdf_text}"
            table_data.append({
                "markdown": table_markdown,
                "combined_content": combined_content,
                "page": table.page
            })
        
        return table_data, pdf_text
    except Exception as e:
        st.error(f"Error extracting tables: {str(e)}")
        return [], pdf_text
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def create_vector_store(table_data):
    if not table_data:
        return None
    documents = [Document(page_content=table["combined_content"], metadata={"page": table["page"]}) for table in table_data]
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    try:
        with st.spinner("Processing PDF..."):
            st.session_state.pdf_data, pdf_text = extract_tables_and_text_from_pdf(uploaded_file)
            st.session_state.vectorstore = create_vector_store(st.session_state.pdf_data or [{"combined_content": pdf_text, "page": 1}])
            st.success("PDF processed successfully! You can now ask questions.")
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on tables and text extracted from a PDF."}
    ]

for msg in st.session_state.conversation_history[1:]:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the PDF"):
    st.session_state.conversation_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.vectorstore:
        with st.chat_message("assistant"):
            try:
                docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
                combined_context = "\n\n".join([doc.page_content for doc in docs])
                response_container = st.empty()
                full_response = ""
                response_stream = client.chat.completions.create(
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
