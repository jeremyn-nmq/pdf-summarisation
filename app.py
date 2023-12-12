import streamlit as st
import os
import torch
from txtai.pipeline import Summary, Textractor
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline

st.set_page_config(layout="wide")

# openai
def load_openai_api_key():
    dotenv_path = ".env"
    load_dotenv(dotenv_path)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(f"Unable to retrieve OPENAI_API_KEY from {dotenv_path}")
    return openai_api_key

@st.cache_resource
def openai_text_summary(text):
    try:
        os.environ["OPENAI_API_KEY"] = load_openai_api_key()
    except ValueError as e:
        st.error(str(e))
        return
    # split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)

    # convert the chunks of text into embeddings to form a knowledge base
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    query = "Summarize the content of the uploaded PDF file in approximately 3-5 sentences. Focus on capturing the main ideas and key points discussed in the document. Use your own words and ensure clarity and coherence in the summary."
    response = ''
    if query:
        docs = knowledgeBase.similarity_search(query)
        OpenAIModel = "gpt-3.5-turbo-16k"
        llm = ChatOpenAI(model=OpenAIModel, temperature=0.1)
        chain = load_qa_chain(llm, chain_type='stuff')

        with get_openai_callback() as cost:
            response = chain.run(input_documents=docs, question=query)
            print(cost)
            return response

# txtai
@st.cache_resource
def txtai_summary(text, maxlength=None):
    #create summary instance
    summary = Summary()
    text = (text)
    return summary(text)

# laMini
@st.cache_resource
def lamini_summary(text):
    checkpoint = "./LaMini-Flan-T5-248M"
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)
    pipe_sum = pipeline(
        'summarization',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 500, 
        min_length = 50)
    result = pipe_sum(text)
    result = result[0]['summary_text']
    return result


# text extraction
@st.cache_resource
def extract_text_from_pdf(file_path):
    # open the PDF file using PyPDF2
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        page = reader.pages[0]
        text = page.extract_text()
    return text

# model calls
def call_models(text):
    st.markdown("*txtai*")
    try:
        st.success(txtai_summary(text))
    except Exception as e:
        st.error(e)

    st.markdown("*openai*")
    try:
        st.success(openai_text_summary(text))
    except Exception as e:
        st.error(e)
    
    st.markdown("*laMini*")
    try:
        st.success(lamini_summary(text))
    except Exception as e:
        st.error(e)

# main function, streamlit app
def main():
    st.title("PDF Summariser")
    st.write("Created by Nguyen Minh Quan - MITIU23204")
    st.divider()

    choice = st.sidebar.selectbox("Select your choice", ["Summarise Text", "Summarise Document"])

    if choice == "Summarise Text":
        st.subheader("Summarise Text using txtai, openai, lamini")
        input_text = st.text_area("Enter your text here")
        if input_text is not None:
            if st.button("Summarize Text"):
                col1, col2 = st.columns([1,1])
                with col1:
                    st.markdown("**Your Input Text**")
                    st.info(input_text)
                with col2:
                    st.markdown("**Summary Result**")
                    call_models(input_text)

    elif choice == "Summarise Document":
        st.subheader("Summarise Document using txtai, openai, lamini")
        input_file = st.file_uploader("Upload your pdf file here", type=['pdf'])
        if input_file is not None:
            if st.button("Summarise Document"):
                with open("doc_file.pdf", "wb") as f:
                    f.write(input_file.getbuffer())
                col1, col2 = st.columns([1,1])
                with col1:
                    st.info("File uploaded successfully")
                    extracted_text = extract_text_from_pdf("doc_file.pdf")
                    st.markdown("**Here is the extracted text from your file:**")
                    st.info(extracted_text)
                with col2:
                    st.markdown("**Summary Result**")
                    text = extract_text_from_pdf("doc_file.pdf")
                    call_models(text)

if __name__ == '__main__':
    main()
