# import os
# import sys
# import json
import boto3
import streamlit as st

# We will be using Titan's Embeddings model to generate embeddings.
# We will call this Titan embeddings from the langchain library.
# Langchain provides you multiple functionalities and options to interact with bedrock.
# We have to learn about Langchain and Lama Index.
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# We also need to import some libraries for data ingestion, we need to load the dataset.
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

# So, we have to convert the loaded PDF documents into vector embeddings.
# Vector Embeddings and Vector Store
# Here, we are using Faiss index here, we can also use Chroma DB
from langchain_community.vectorstores import FAISS

# LLM Models
# Langchain already provides ways to load models in the AWS bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock Clients
# Here, we can access the models.
bedrock_client = boto3.client(service_name='bedrock-runtime')

# You can see the model id in AWS Bedrock service. Also, you can see the API request model.
# Here, we are going to use Titan Embedding model.
model_id = "amazon.titan-embed-text-v1"
bedrock_embeddings = BedrockEmbeddings(model_id=model_id, client=bedrock_client)


# STEP 1 - Creating the Data Ingestion Model.
def data_ingestion():

    # Here, we are reading all the PDFs from the "data" folder.
    loader = PyPDFDirectoryLoader('data')
    documents = loader.load()

    # Based on the Recursive Character Text Splitter, we are splitting our documents
    # This "RecursiveCharacterTextSplitter" works well in splitting the documents.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)

    docs = text_splitter.split_documents(documents)

    return docs


# Vector Embedding and Vector Store.
# Here, we are specifically using the Titan Embedding imported along with that we are going to specifically use the FIAS
def get_vector_store(docs):

    # At first, we are taking the documents from the ingested place
    # Then, we are doing vector embeddings.
    vector_store_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )

    # We will be storing those documents in the local disk.
    # This, we will be storing in folder over here.
    vector_store_faiss.save_local("faiss_index")


# Based on the import, we have to work with LLM models.
# So, we are creating LLM models here.
def get_claude_llm():

    # Create the Anthropic model. Already, Bedrock gives us the power to use the multiple models.
    llm1 = Bedrock(model_id="ai21.j2-mid-v1", client=bedrock_client,
                   model_kwargs={'maxTokens': 512})

    return llm1


def get_llama2_llm():

    llm2 = Bedrock(model_id="ai21.j2-mid-v1", client=bedrock_client,
                   model_kwargs={'max_gen_length': 512})

    return llm2


prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the 
end but use atleast summarize 250 words with detailed explanations. If you don't know the answer,
just say that you don't know, don't try to makeup the answer. 
<context>
{context}
</context

Question: {question}

Assistant: """

prompt = PromptTemplate(
    template=prompt_template, input_variables=['context', 'question']
)


def get_response_llm(llm, vector_store_faiss, query):

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        # This is the important place, it will say where the similarity happens.
        # Vector store Faiss has an entire index itself.
        retriever=vector_store_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

    answer = qa({'query': query})['result']       # Query is coming as an input.

    return answer


def main():

    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bed Rock")

    user_question = st.text_input("Ask a question from the PDF files.")

    with st.sidebar:
        st.title("Update or Create vector store:")

        # As soon as we click the vector update button, the vector stores will be created.
        if st.button("Vectors Update"):
            with st.spinner('Processing...'):

                # This function is going to read all the functions from the data folder.
                docs = data_ingestion()

                # Now, we are storing the documents in the hard-disk in the form of faiss index.
                get_vector_store(docs)

                st.success('Done')

        # As soon as we click the claude button, our output should be loaded from the local.
        if st.button("Claude Button"):
            with st.spinner('Processing...'):

                faiss_index = FAISS.load_local('faiss_index', bedrock_embeddings)
                llm = get_claude_llm()

                st.write(get_response_llm(llm, faiss_index, user_question))
                st.success('Done')


# Run this program using the command, "streamlit run app.py"
if __name__ == '__main__':
    main()
