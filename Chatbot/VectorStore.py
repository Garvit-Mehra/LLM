from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class VectorStoreManager:
    def __init__(self):
        self.vectorstore = None

    def create_vector_store(self, table_data, pdf_text):
        """Create FAISS vector store from table data."""
        if not table_data:
            table_data = [{"combined_content": pdf_text, "page": 1}]
        documents = [Document(page_content=table["combined_content"], metadata={"page": table["page"]}) 
                    for table in table_data]
        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_documents(documents, embeddings)
        return self.vectorstore

    def search(self, query, k=3):
        """Search vector store for relevant documents."""
        if self.vectorstore:
            return self.vectorstore.similarity_search(query, k=k)
        return []