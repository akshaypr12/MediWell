import os
import faiss
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from dotenv import load_dotenv
from transformers import BertTokenizer, BertModel
import torch
import ollama
import streamlit as st

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
@st.cache_resource
def load_data():
    """Loads the dataset and pre-computed embeddings."""
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset not found at '{DATASET_PATH}'!")
        st.stop()

    df = pd.read_excel(DATASET_PATH)

    if not os.path.exists(EMBEDDINGS_PATH):
        st.error(f"Embeddings file not found at '{EMBEDDINGS_PATH}'!")
        st.stop()

    doctor_embeddings = np.load(EMBEDDINGS_PATH)
    return df, doctor_embeddings

# Load FAISS index
@st.cache_resource
def load_faiss_index():
    """Loads the FAISS index from disk."""
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error(f"FAISS index not found at '{FAISS_INDEX_PATH}'!")
        st.stop()

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

# Search in FAISS
def search_similar_responses(query, model, tokenizer, device, index, df, top_k=5, min_similarity=0.80):
    """Searches for similar responses in FAISS using cosine similarity."""
    query_embedding = get_embeddings([query], model, tokenizer, device)[0]
    query_embedding_normalized = normalize(query_embedding.reshape(1, -1), axis=1)

    distances, indices = index.search(query_embedding_normalized.astype(np.float32), top_k)

    results = []
    low_confidence_results = []

    for idx, score in zip(indices[0], distances[0]):
        response = df.iloc[idx]["Doctor_pp"]
        if score >= min_similarity:
            results.append((response, score))
        else:
            low_confidence_results.append((response, score))

    return results, low_confidence_results

# Generate response using Ollama
def generate_response_ollama(query, retrieved_docs, model_name="mistral"):
    """Generates a refined RAG response using Ollama LLM locally."""
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
        return f"No response generated due to an error: {e}"

# Streamlit UI
def main():
    st.title("Medical Chatbot with FAISS and Ollama")
    st.write("Ask a medical question and get responses based on FAISS + RAG.")

    # Load model, data, and FAISS index
    with st.spinner("Loading model, data, and FAISS index..."):
        model, tokenizer, device = load_model()
        df, _ = load_data()
        index = load_faiss_index()

    query = st.text_area("Enter your medical query:")

    min_similarity = st.slider("Minimum Similarity Threshold", 0.0, 1.0, 0.80, 0.01)
    top_k = st.slider("Top K Results", 1, 10, 5)

    if st.button("Search"):
        if query.strip():
            with st.spinner("Retrieving similar responses..."):
                similar_responses, low_confidence = search_similar_responses(
                    query, model, tokenizer, device, index, df, top_k, min_similarity
                )

            if not similar_responses and not low_confidence:
                st.warning("No relevant responses found.")
            else:
                # Display retrieved responses
                st.subheader("Top Retrieved Responses")
                for i, (response, score) in enumerate(similar_responses):
                    st.write(f"{i + 1}. {response} (Similarity: {score:.4f})")

                if low_confidence:
                    st.subheader("Low-Confidence Matches")
                    for i, (response, score) in enumerate(low_confidence):
                        st.write(f"{i + 1}. {response} (Similarity: {score:.4f})")

                # Generate RAG response
                with st.spinner("Generating RAG response with Ollama..."):
                    rag_response = generate_response_ollama(query, similar_responses)

                st.subheader("RAG-Generated Response")
                st.write(rag_response)

        else:
            st.warning("Please enter a query before searching.")

# Run the app
if __name__ == "__main__":
    main()
