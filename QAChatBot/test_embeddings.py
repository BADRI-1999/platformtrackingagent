from sentence_transformers import SentenceTransformer
import numpy as np

def test_embeddings():
    # Initialize the model
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    
    # Test sentences
    sentences = [
        "What are the legal requirements?",
        "What is the policy regarding data privacy?",
        "How to handle compliance issues?",
        "What are the regulatory guidelines?"
    ]
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(sentences)
    
    # Calculate similarities
    similarities = np.inner(embeddings, embeddings)
    
    print("\nSimilarity Matrix:")
    for i, sent1 in enumerate(sentences):
        for j, sent2 in enumerate(sentences):
            print(f"\nSimilarity between:\n'{sent1}' and\n'{sent2}':\n{similarities[i][j]:.4f}")
    
    return embeddings, similarities

if __name__ == "__main__":
    embeddings, similarities = test_embeddings() 