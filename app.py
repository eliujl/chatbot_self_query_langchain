# Import required libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    UnstructuredWordDocumentLoader,
    PyMuPDFLoader,
    UnstructuredFileLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
import os
import langchain
import pinecone
import streamlit as st
import shutil
import json
import re

OPENAI_API_KEY = ''
PINECONE_API_KEY = ''
PINECONE_API_ENV = ''
langchain.verbose = False

@st.cache_data()
def init():
    pinecone_index_name = ''
    pinecone_namespace = ''
    docsearch_ready = False
    directory_name = 'tmp_docs'
    return pinecone_index_name, pinecone_namespace, docsearch_ready, directory_name


@st.cache_data()
def save_file(files):
    # Remove existing files in the directory
    if os.path.exists(directory_name):
        for filename in os.listdir(directory_name):
            file_path = os.path.join(directory_name, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error: {e}")
    # Save the new file with original filename
    if files is not None:
        for file in files:
            file_name = file.name
            file_path = os.path.join(directory_name, file_name)
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(file, f)


def load_files():
    all_texts = []
    n_files = 0
    n_char = 0
    n_texts = 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50
    )
    for filename in os.listdir(directory_name):
        file = os.path.join(directory_name, filename)
        if os.path.isfile(file):
            if file.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file)
            elif file.endswith(".pdf"):
                loader = PyMuPDFLoader(file)
            else:   # assume a pure text format and attempt to load it
                loader = UnstructuredFileLoader(file)
            data = loader.load()
            metadata = data[0].metadata
            fn = os.path.basename(metadata['source'])
            author = os.path.splitext(fn)[0]
            data[0].metadata['author'] = author
            texts = text_splitter.split_documents(data)
            n_files += 1
            n_char += len(data[0].page_content)
            n_texts += len(texts)
            all_texts.extend(texts)
    st.write(
        f"Loaded {n_files} file(s) with {n_char} characters, and split into {n_texts} split-documents."
    )
    return all_texts, n_texts


@st.cache_resource()
def ingest(_all_texts, _embeddings, pinecone_index_name, pinecone_namespace):
	docsearch = Pinecone.from_documents(
            _all_texts, _embeddings, index_name=pinecone_index_name, namespace=pinecone_namespace)
	return docsearch


def setup_retriever(docsearch, llm):
    metadata_field_info = [
        AttributeInfo(
            name="author",
            description="The author of the document/text/piece of context",
            type="string or list[string]",
        )
    ]
    document_content_description = "Views/opions/proposals suggested by the author on one or more discussion points."
    retriever = SelfQueryRetriever.from_llm(
        llm, docsearch, document_content_description, metadata_field_info, verbose=True)
    return retriever


def setup_docsearch(pinecone_index_name, pinecone_namespace, embeddings):
    docsearch = []
    n_texts = 0
	# Load the pre-created Pinecone index.
	# The index which has already be stored in pinecone.io as long-term memory
    if pinecone_index_name in pinecone.list_indexes():
        docsearch = Pinecone.from_existing_index(
            index_name=pinecone_index_name, embedding=embeddings, text_key='text', namespace=pinecone_namespace)
        index_client = pinecone.Index(pinecone_index_name)
		# Get the index information
        index_info = index_client.describe_index_stats()
        n_texts = index_info['namespaces'][pinecone_namespace]['vector_count']
    else:
        raise ValueError('''Cannot find the specified Pinecone index.
						Create one in pinecone.io or using, e.g.,
						pinecone.create_index(
							name=index_name, dimension=1536, metric="cosine", shards=1)''')
    return docsearch, n_texts


def get_response(query, chat_history, CRqa):
    result = CRqa({"question": query, "chat_history": chat_history})
    return result['answer'], result['source_documents']


def setup_em_llm(OPENAI_API_KEY, temperature):
    # Set up OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # Use Open AI LLM with gpt-3.5-turbo.
    # Set the temperature to be 0 if you do not want it to make up things
    llm = ChatOpenAI(temperature=temperature, model_name="gpt-3.5-turbo", streaming=True,
                     openai_api_key=OPENAI_API_KEY)
    return embeddings, llm


def load_chat_history(CHAT_HISTORY_FILENAME):
    try:
        with open(CHAT_HISTORY_FILENAME, 'r') as f:
            chat_history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        chat_history = []
    return chat_history


def save_chat_history(chat_history, CHAT_HISTORY_FILENAME):
    with open(CHAT_HISTORY_FILENAME, 'w') as f:
        json.dump(chat_history, f)


pinecone_index_name, pinecone_namespace, docsearch_ready, directory_name = init()


def main(pinecone_index_name, pinecone_namespace, docsearch_ready):
    docsearch_ready = False
    chat_history = []
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        r_ingest = st.radio(
            'Ingest file(s)?', ('Yes', 'No'))
        OPENAI_API_KEY = st.text_input(
            "OpenAI API key:", type="password")

    with col2:        
        PINECONE_API_KEY = st.text_input(
				"Pinecone API key:", type="password")
        PINECONE_API_ENV = st.text_input(
				"Pinecone API env:", type="password")
        pinecone_index_name = st.text_input('Pinecone index:')
        pinecone.init(api_key=PINECONE_API_KEY,
							environment=PINECONE_API_ENV)
    with col3:
        pinecone_namespace = st.text_input('Pinecone namespace:')
        temperature = st.slider('Temperature', 0.0, 1.0, 0.1)
    
    if pinecone_index_name:
        session_name = pinecone_index_name
        embeddings, llm = setup_em_llm(OPENAI_API_KEY, temperature)
        if r_ingest.lower() == 'yes':
            files = st.file_uploader(
                'Upload Files', accept_multiple_files=True)
            if files:
                save_file(files)
                all_texts, n_texts = load_files()
                docsearch = ingest(all_texts, embeddings,
                                   pinecone_index_name, pinecone_namespace)
                docsearch_ready = True
        else:
            st.write(
                'No data is to be ingested. Make sure the Pinecone index you provided contains data.')
            docsearch, n_texts = setup_docsearch(pinecone_index_name, pinecone_namespace, 
                                                 embeddings)
            docsearch_ready = True
    if docsearch_ready:
        retriever = setup_retriever(docsearch, llm)
        CRqa = load_qa_with_sources_chain(llm, chain_type="stuff")

        st.title('Chatbot')
        # Get user input
        query = st.text_area('Enter your question:', height=10,
                             placeholder='Summarize the context.')
        if query:
            # Generate a reply based on the user input and chat history
            CHAT_HISTORY_FILENAME = f"chat_history/{session_name}_chat_hist.json"
            chat_history = load_chat_history(CHAT_HISTORY_FILENAME)
            chat_history = [(user, bot)
                            for user, bot in chat_history]
            docs = retriever.get_relevant_documents(query)
            if not docs:
                docs = docsearch.similarity_search(query)
            result = CRqa.run(input_documents=docs, question=query)
            reply = re.match(r'(.+?)\.\s*SOURCES:', result).group(1)
            source = re.search(r'SOURCES:\s*(.+)', result).group(1)
            # Update the chat history with the user input and system response
            chat_history.append(('User', query))
            chat_history.append(('Bot', reply))
            save_chat_history(chat_history, CHAT_HISTORY_FILENAME)
            latest_chats = chat_history[-4:]
            chat_history_str = '\n'.join(
                [f'{x[0]}: {x[1]}' for x in latest_chats])
            st.text_area('Chat record:', value=chat_history_str, height=250)


if __name__ == '__main__':
    main(pinecone_index_name, pinecone_namespace, 
         docsearch_ready)
