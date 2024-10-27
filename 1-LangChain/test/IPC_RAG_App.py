from langchain.document_loaders import PyPDFLoader
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import streamlit as st 

LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT="basic_project_model_io"
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")  
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_TRACING_V2=True

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:            text += page.extract_text()
    return text

def get_text_chunks(text):
  text_splitter = CharacterTextSplitter(
      separator="\n",
      chunk_size=1000,
      chunk_overlap=200,
      length_function=len
  )
  chunks = text_splitter.split_text(text)
  return chunks

def get_embeddings(text_chunks):
  embeddings = OpenAIEmbeddings()
  return embeddings

def get_convesrational_chain(vectorstore):
  llm = ChatOpenAI(temperature=0)
  memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
  convesrational_chain = ConversationalRetrievalChain.from_llm(
      llm=llm,
      retriever=vectorstore.as_retriever(),
      memory=memory
  )
  return convesrational_chain

def handle_userinput(user_question):
  response = st.session_state.convesrational_chain({"question": user_question})
  st.session_state.chat_history = response['chat_history']

  for i, message in enumerate(st.session_state.chat_history):
    if i % 2 == 0 :
      st.write(':man_in_tuxedo:', message.content)
    else:
      st.write(':robot_face:', message.content)

def main():

    st.set_page_config(page_title="Chat with Documents", page_icon=":books:")
    styl = f"""
      <style>
          .stTextInput {{
            position: fixed;
            bottom: 3rem;
          }}
      </style>
      """
    st.markdown(styl, unsafe_allow_html=True)

    #st.write(css, unsafe_allow_html=True)
    st.title("AI Chat-Bot for Your Documents :books:")

    if "convesrational_chain" not in st.session_state:
        st.session_state.convesrational_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    user_input=st.text_input("Upload files, Hit Submit button, then ask questions.:blue[- INDIA -]:sunglasses:")
    if user_input:
      handle_userinput(user_input)


    with st.sidebar:
        pdf_docs=st.file_uploader("Upload your documents here" , accept_multiple_files=True)
        if st.button("Submit"):
          with st.spinner("Processing..."):

            # Get PDF Data
            raw_text=get_pdf_text(pdf_docs)

            #Divide the Data into Chunks
            chunked_array=get_text_chunks(raw_text)

            #Create embeddings
            embeddings=get_embeddings(chunked_array)

            #Create vectorstore
            vectorstore=FAISS.from_texts(chunked_array,embeddings)

            #crate a converstaion chain
            st.session_state.convesrational_chain=get_convesrational_chain(vectorstore)



if __name__ == "__main__":
    main()