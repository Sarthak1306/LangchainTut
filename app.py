from dotenv import load_dotenv
import os
import streamlit as st
import pinecone
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

# from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Pinecone


def main():
    load_dotenv()  # Loading .env file
    st.set_page_config(
        page_title="Chat with PDF"
    )  # using Streamlit to create a basic webapp
    st.header("Chat with PDF's 👀")

    # Uploading the PDF File
    pdf = st.file_uploader(
        "Upload your PDF", type="pdf"
    )  # streamlit file uploader function

    # Extracting the File Text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split into chunks, still inside the if pdf
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text)

        pinecone.init(
            api_key="d47e1374-e2a7-40ee-a2f4-b47de1fa5272",
            environment="asia-southeast1-gcp-free",
        )

        # Create Embeddings
        embeddings = OpenAIEmbeddings()
        index_name = "langchaintut"
        # knowledge_base = FAISS.from_texts(chunks,embeddings) //FAISS
        vectorstore = Pinecone.from_texts(chunks, embeddings, index_name=index_name)

        # Show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = vectorstore.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            st.write(response)


if __name__ == "__main__":
    main()
