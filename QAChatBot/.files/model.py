from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from sentence_transformers import SentenceTransformer
import requests
import os



DB_FAISS_PATH = "//content//drive//MyDrive//QAChatBot//LawDocbase"


print("DB_FAISS_PATH = ", DB_FAISS_PATH)



custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 8}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Llama-2-13B-chat-GGML",
        model_type="llama",
        max_new_tokens = 2000,
        temperature = 0.5,
        config= {
            'context_length':2000
        }
    )
    

    # llm = ChatOpenAI(
    # model_name="gpt-3.5-turbo",
    # temperature=0,
    # max_tokens=2000,
    # model_kwargs = {
    #     'frequency_penalty':0,
    #     'presence_penalty':0,
    #     'top_p':1.0
    # }
    # )
    
    return llm

#QA Model Function
def qa_bot():
    
    # DB_FAISS_PATH = os.path.join('API','uploads',botid,'Data','db','db_faiss')

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                       model_kwargs={'device': 'cuda'})
   

   
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    faiss_index_dimension = db.index.d

    print("Faiss index dimension:", faiss_index_dimension)

    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa



# Process Query Function
def process_query(query):
    qa = qa_bot()
    response = qa({'query': query})
    return response

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Query the QA bot from the command line.")
    parser.add_argument("query", type=str, help="The question to ask the QA bot.")
    args = parser.parse_args()

    # Get the query from command line argument
    query = args.query
    print("Query:", query)

    # Get the response
    response = process_query(query)
    print("Response:", response["result"])
    if response.get("source_documents"):
        print("Sources:")
        for doc in response["source_documents"]:
            print(f"- {doc.metadata.get('source', 'unknown')}")

