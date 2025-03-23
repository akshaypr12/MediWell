import streamlit as st
import os
import pandas as pd
import faiss
import numpy as np
import torch
import ollama

from dotenv import load_dotenv
from sklearn.preprocessing import normalize
from transformers import BertTokenizer, BertModel

# Load environment variables
load_dotenv()

# Constants
DATASET_PATH = "Cleaned_data.xlsx"
FAISS_INDEX_PATH = "faiss_index.bin"
EMBEDDINGS_PATH = "doctor_embeddings.npy"

# Load Bio_ClinicalBERT model
@st.cache_resource
def load_model():
    """Loads the Bio_ClinicalBERT model and tokenizer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)

    return model, tokenizer, device

# Load dataset and embeddings
@st.cache_data
def load_data():
    """Loads dataset and embeddings."""
    df = pd.read_excel(DATASET_PATH)
    doctor_embeddings = np.load(EMBEDDINGS_PATH)
    return df, doctor_embeddings

# Load FAISS index
@st.cache_resource
def load_faiss_index():
    """Loads the FAISS index."""
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at '{FAISS_INDEX_PATH}'!")
    
    index = faiss.read_index(FAISS_INDEX_PATH)
    return index

# Generate embeddings
def get_embeddings(texts, model, tokenizer, device, batch_size=16):
    """Generates embeddings using Bio_ClinicalBERT."""
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)

# Search for similar responses
def search_similar_responses(query, model, tokenizer, device, index, df, top_k=5, min_similarity=0.80):
    """Searches FAISS for similar responses."""
    query_embedding = get_embeddings([query], model, tokenizer, device)[0]
    query_embedding_normalized = normalize(query_embedding.reshape(1, -1), axis=1)

    distances, indices = index.search(query_embedding_normalized.astype(np.float32), top_k)

    results = []
    low_confidence = []

    for idx, score in zip(indices[0], distances[0]):
        response = df.iloc[idx]["Doctor_pp"]

        if score >= min_similarity:
            results.append((response, score))
        else:
            low_confidence.append((response, score))

    return results, low_confidence

# Generate RAG response using Ollama
def generate_response_ollama(query, retrieved_docs, model_name="mistral"):
    """Generates a refined response using Ollama."""
    context = "\n".join([f"- {doc}" for doc, _ in retrieved_docs])

    prompt = f"""
You are a highly skilled and empathetic medical assistant. Use the retrieved information to answer the patient's question thoroughly and accurately.

Patient Question:
{query}

Retrieved Information:
{context}

Please provide a detailed, empathetic, and clear response, incorporating the relevant medical context where appropriate.
"""

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response.get('message', {}).get('content', "No response generated.")
    
    except Exception as e:
        return f"Error: {e}"

# Streamlit interface
def main():
    st.title("Medical Chatbot")

    # Load resources
    model, tokenizer, device = load_model()
    df, embeddings = load_data()
    index = load_faiss_index()

    query = st.text_area("Enter your medical question:")

    min_similarity = st.slider("Minimum Similarity Threshold", 0.5, 1.0, 0.80, 0.05)
    top_k = st.slider("Top K Results", 1, 10, 5)

    if st.button("Get Response"):
        if query:
            with st.spinner("Searching and generating response..."):
                # Retrieve similar responses
                similar_responses, low_confidence = search_similar_responses(
                    query, model, tokenizer, device, index, df, top_k, min_similarity
                )

                if similar_responses:
                    # Generate RAG response
                    rag_response = generate_response_ollama(query, similar_responses)

                    # Display results
                    st.subheader("RAG-Generated Response:")
                    st.write(rag_response)

                    # Display retrieved documents
                    st.subheader("Retrieved Documents:")
                    for i, (doc, score) in enumerate(similar_responses):
                        st.write(f"{i + 1}. {doc}  \n(Similarity: {score:.4f})")

                    # Display low-confidence results
                    if low_confidence:
                        st.subheader("Low-Confidence Matches:")
                        for i, (doc, score) in enumerate(low_confidence):
                            st.write(f"{i + 1}. {doc}  \n(Similarity: {score:.4f})")

                else:
                    st.warning("No relevant responses found.")
        else:
            st.error("Please enter a query!")

# Run the Streamlit app
if __name__ == "__main__":
    main()
