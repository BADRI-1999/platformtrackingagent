import argparse
from transformers import AutoTokenizer, LlamaForCausalLM
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
import torch

DB_FAISS_PATH = "/content/drive/MyDrive/QAChatBot/LawDocbase/db_faiss"

def load_model_and_tokenizer(model_name):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlamaForCausalLM.from_pretrained(model_name).to(device)
    
    return tokenizer, model, device

def retrieve_context(query, embeddings_model, db_faiss_path, top_k=5):
    # Load the FAISS vector database
    db = FAISS.load_local(db_faiss_path, embeddings_model, allow_dangerous_deserialization=True)
    
    # Perform similarity search using the retriever
    docs = db.similarity_search(query, k=top_k)
    context = "\n".join([doc.page_content for doc in docs])
    return context

def generate_response(model, tokenizer, query, context, max_length=200):
    # Combine the context and query to form the prompt
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate a response
    generate_ids = model.generate(inputs.input_ids, max_length=max_length)
    
    # Decode the generated tokens to text
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response

def main():
    parser = argparse.ArgumentParser(description="Generate text using Llama-2 model with FAISS vector database.")
    parser.add_argument("query", type=str, help="The input query for the model.")
    parser.add_argument("--model_name", type=str, default="facebook/llm-compiler-13b", help="Name of the model to use.")
    parser.add_argument("--db_faiss_path", type=str, default=DB_FAISS_PATH, help="Path to the FAISS database.")
    parser.add_argument("--max_length", type=int, default=2000, help="Maximum length of the generated response.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top documents to retrieve for context.")
    args = parser.parse_args()

    print("Loading model and tokenizer...")
    tokenizer, model = load_model_and_tokenizer(args.model_name)

    print("Loading embeddings model...")
    # Update the import as per the deprecation warning
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    print("Retrieving context...")
    context = retrieve_context(args.query, embeddings_model, args.db_faiss_path, args.top_k)

    print("Generating response...")
    response = generate_response(model, tokenizer, args.query, context, args.max_length)

    print("Response:\n", response)

if __name__ == "__main__":
    main()
