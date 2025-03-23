# %%
#Load data
from datasets import load_dataset

# Load the dataset
ds = load_dataset("sid6i7/patient-doctor")
print(ds)

# %%
import pandas as pd
df = ds["train"].to_pandas()
df.head()

# %%
columns_to_drop = ['Unnamed: 0', 'problem_description']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
df.head(5)

# %%
df.isna().sum()

# %%
df = df.dropna(subset=['Patient'])
df.isna().sum()

# %%
import re
import pandas as pd

# Define the preprocessing function
def preprocess_text(text, remove_words=None):
    """
    Preprocess text by removing placeholders, specific words, extra spaces,
    non-alphanumeric characters, and normalizing text to lowercase.
    """
    if not isinstance(text, str):
        return text  
    
    # Define words to remove (default or custom)
    remove_words = remove_words or ['<start>', 'XXXX', 'hello', 'hi', 'start', 'thanks', 'Dear sir']
    
    # Remove specified words or placeholders
    for word in remove_words:
        text = re.sub(fr'\b{re.escape(word)}\b', '', text, flags=re.IGNORECASE)
    
    # Remove non-alphanumeric characters (except punctuation)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'":;]', '', text)
    
    # Remove extra spaces and normalize text
    text = re.sub(r'\s+', ' ', text).strip().lower()
    
    return text


df['Description_pp'] = df['Description'].apply(preprocess_text)
df['Patient_pp'] = df['Patient'].apply(preprocess_text)
df['Doctor_pp'] = df['Doctor'].apply(preprocess_text)

# %%
df = df[['Description_pp','Patient_pp', 'Doctor_pp']]
df.head()
df.to_excel('Cleaned_data.xlsx', index=False)
# %%
#pip install faiss-cpu

# %%
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel


# Load Model and Tokenizer
def load_model():
    """
    Load the Bio_ClinicalBERT model and tokenizer.
    """
    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    return model, tokenizer, device


# Embedding Generation Function
def get_embeddings(texts, model, tokenizer, device, batch_size=16, max_length=128):
    """
    Generate embeddings using Bio_ClinicalBERT.

    Args:
    - texts (list): List of text strings.
    - model: Bio_ClinicalBERT model instance.
    - tokenizer: Tokenizer instance.
    - device (str): "cuda" or "cpu".
    - batch_size (int): Batch size for processing.
    - max_length (int): Max length for tokenization.

    Returns:
    - np.ndarray: Array of embeddings.
    """
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
        batch = texts[i: i + batch_size]

        # Tokenize and move to device
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract CLS token embeddings
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)

    # Combine all batches into a single numpy array
    return np.vstack(all_embeddings).astype(np.float32)


# Main Execution Function
def main():
    """
    Main function to load the dataset, generate embeddings, and save them.
    """
    # Load model and tokenizer
    model, tokenizer, device = load_model()

    # Load the dataset
    dataset_path = "Cleaned_data.xlsx"  # Replace with your dataset path
    print(f"\nLoading dataset from: {dataset_path}")

    try:
        df = pd.read_excel(dataset_path)  # Use read_csv() if your dataset is in CSV format
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Check if the required column exists
    if "Doctor_pp" not in df.columns:
        raise KeyError("Dataset must contain 'Doctor_pp' column.")

    # Generate embeddings
    print("\nGenerating embeddings...")
    doctor_embeddings = get_embeddings(
        texts=df["Doctor_pp"].tolist(),
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    # Normalize embeddings for cosine similarity
    doctor_embeddings_normalized = normalize(doctor_embeddings, axis=1)

    # Save embeddings to file
    output_path = "doctor_embeddings.npy"
    np.save(output_path, doctor_embeddings_normalized)
    print(f"\nEmbeddings saved successfully at '{output_path}'!")


# Execute the main function only when running this script directly
if __name__ == "__main__":
    main()
