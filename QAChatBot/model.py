import argparse
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd
import json
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings

# Database configuration
DB_URL = "sqlite:///qa_database.db"
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Vector store configuration
DB_FAISS_PATH = "LawDocbase/db_faiss"
faiss_index_file = os.path.join(DB_FAISS_PATH, "index.faiss")
print("DB_FAISS_PATH =", DB_FAISS_PATH)
print("Final FAISS Index File Path:", faiss_index_file)

class SimpleEmbeddings:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
    def embed_documents(self, texts):
        """Embed a list of texts."""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            print(f"Error in embed_documents: {e}")
            raise
    
    def embed_query(self, text):
        """Embed a single text query."""
        try:
            embedding = self.model.encode([text], convert_to_tensor=False)[0]
            return embedding.tolist()
        except Exception as e:
            print(f"Error in embed_query: {e}")
            raise

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def query_database(question, db_session):
    """
    Query the SQL database for relevant information with improved matching
    """
    try:
        # First try exact match
        query = text("""
            SELECT response, context, question
            FROM qa_pairs
            WHERE LOWER(question) = LOWER(:exact_question)
            LIMIT 1
        """)
        result = db_session.execute(query, {"exact_question": question}).fetchone()
        
        if result:
            print(f"Found exact match for question: {result.question}")
            return result

        # If no exact match, try fuzzy match using LIKE
        words = question.lower().split()
        if len(words) > 2:
            # Create a query that matches any of the significant words
            like_conditions = []
            params = {}
            for i, word in enumerate(words):
                if len(word) > 3:  # Only use words longer than 3 characters
                    param_name = f"word_{i}"
                    like_conditions.append(f"LOWER(question) LIKE :%{param_name}%")
                    params[param_name] = f"%{word}%"
            
            if like_conditions:
                query = text(f"""
                    SELECT response, context, question
                    FROM qa_pairs
                    WHERE {' OR '.join(like_conditions)}
                    ORDER BY last_updated DESC
                    LIMIT 1
                """)
                result = db_session.execute(query, params).fetchone()
                if result:
                    print(f"Found fuzzy match for question: {result.question}")
                    return result

        print("No matching question found in database")
        return None
    except Exception as e:
        print(f"Database query error: {e}")
        return None

custom_prompt_template = """
You are a legal expert. Use the following context to answer the user's question about the law. Provide both a concise summary and a detailed explanation.

Context: {context}
Question: {question}

Respond with two parts:
1. **Short Answer**: A brief summary of the answer in 1-2 sentences.
2. **Detailed Answer**: A more comprehensive explanation of answer in minimum 10 sentences and maximum of 15 sentences.

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

    
def load_llm():
    print("Loading the model...")
    try:
        # Use local model from models folder
        model_path = 'models';
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        print(f"Loading model from: {model_path}")
        llm = CTransformers(
            model=model_path,
            model_type="mistral",
            max_new_tokens=512,
            temperature=0.7,
            config={'context_length': 2048}
        )
        
        print("Model loaded successfully.")
        return llm
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

class ModelContextManager:
    def __init__(self, llm, vector_store, db_session):
        self.llm = llm
        self.vector_store = vector_store
        self.db_session = db_session
        self.context_history = []

    def get_real_time_data(self, question):
        """
        Query the database for real-time data
        """
        try:
            query = text("""
                SELECT response, context, last_updated
                FROM qa_pairs
                WHERE question LIKE :question
                AND last_updated >= datetime('now', '-1 hour')
                ORDER BY last_updated DESC
                LIMIT 1
            """)
            result = self.db_session.execute(query, {"question": f"%{question}%"}).fetchone()
            return result if result else None
        except Exception as e:
            print(f"Real-time database query error: {e}")
            return None

    def get_vector_store_data(self, question):
        """
        Get relevant information from vector store
        """
        try:
            docs = self.vector_store.similarity_search(question, k=3)
            return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        except Exception as e:
            print(f"Vector store query error: {e}")
            return []

    def combine_contexts(self, real_time_data, vector_data):
        """
        Implement MCP to combine different context sources
        """
        combined_context = {
            "real_time": real_time_data if real_time_data else None,
            "vector_store": vector_data if vector_data else [],
            "timestamp": datetime.now().isoformat()
        }
        self.context_history.append(combined_context)
        return combined_context

    def generate_response(self, question, combined_context):
        """
        Generate response using combined context
        """
        # Format context for the model
        context_text = ""
        if combined_context["real_time"]:
            context_text += f"Real-time information:\n{combined_context['real_time'].context}\n\n"
        
        if combined_context["vector_store"]:
            context_text += "Historical information:\n"
            for doc in combined_context["vector_store"]:
                context_text += f"{doc['content']}\n"

        # If no context is available, provide a default response
        if not context_text.strip():
            return {
                "answer": "I apologize, but I don't have any information about sensors in apt 101 in my current knowledge base. "
                         "Please make sure the relevant information is added to either the vector store or the database.",
                "source": "default"
            }

        prompt = PromptTemplate(
            template=custom_prompt_template,
            input_variables=['context', 'question']
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=self.vector_store.as_retriever(search_kwargs={'k': 3}),
            chain_type_kwargs={'prompt': prompt}
        )
        
        try:
            # Use invoke instead of run
            response = chain.invoke({
                "query": question,
                "context": context_text
            })
            
            if isinstance(response, dict) and "result" in response:
                return response["result"]
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question. "
                         "Please try rephrasing your question or ensure the relevant information is available.",
                "error": str(e)
            }

def qa_bot():
    try:
        print("Initializing embeddings model...")
        LOCAL_MODEL_PATH = "models\sentenceTransformers"

        embeddings = HuggingFaceEmbeddings(
            model_name=LOCAL_MODEL_PATH,
            model_kwargs={"device": "cpu"}  # or "cuda" if you have a GPU
        )
        
        print("Loading FAISS database...")
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        
        print("Loading LLM...")
        llm = load_llm()
        print("model loaded, back to qa function")
        
        return db, llm
    except Exception as e:
        print(f"Error initializing QA bot: {e}")
        raise

def process_query(query):
    db_session = next(get_db())
    try:
        vector_store, llm = qa_bot()
        print("qa_bot function done")
        context_manager = ModelContextManager(llm, vector_store, db_session)
        
        # Get data from both sources
        real_time_data = context_manager.get_real_time_data(query)
        vector_data = context_manager.get_vector_store_data(query)
        
        if not real_time_data and not vector_data:
            return {
                "result": "I don't have any information about that in my knowledge base. "
                         "Please ensure the relevant data is added to either the vector store or the database.",
                "context_sources": {
                    "real_time_used": False,
                    "vector_store_used": False,
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        # Combine contexts using MCP
        combined_context = context_manager.combine_contexts(real_time_data, vector_data)
        
        # Generate response
        response = context_manager.generate_response(query, combined_context)
        
        return {
            "result": response if isinstance(response, str) else response.get("answer", "No answer generated"),
            "context_sources": {
                "real_time_used": bool(real_time_data),
                "vector_store_used": bool(vector_data),
                "timestamp": combined_context["timestamp"]
            }
        }
    finally:
        db_session.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the QA bot from the command line.")
    parser.add_argument("query", type=str, help="The question to ask the QA bot.")
    args = parser.parse_args()

    query = args.query
    print("\nQuery:", query)
    print("\nSearching knowledge base...")

    response = process_query(query)
    print("\nResponse:", response["result"])
    print("\nSource information:", response["context_sources"])

