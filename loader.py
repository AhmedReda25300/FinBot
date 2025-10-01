import json
import os
import pickle
import google.generativeai as genai
import faiss
import numpy as np
import hashlib
from dotenv import load_dotenv
load_dotenv()

def get_embedding(text, api_key, task_type="retrieval_document"):
    """
    Get embedding for the given text using Google embeddings model.
    """
    genai.configure(api_key=api_key)
    result = genai.embed_content(
        model="gemini-embedding-001",
        content=text,
        task_type=task_type
    )
    return np.array(result['embedding'], dtype=np.float32)

def split_text(text, max_tokens=2048, overlap_ratio=0.1):
    """
    Split text into smaller chunks if it exceeds max_tokens, with overlap.
    Returns a list of text chunks.
    """
    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    chunks = []
    start = 0
    overlap = int(max_tokens * overlap_ratio)
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap if overlap > 0 else end

    return chunks

def generate_unique_id(item, filename):
    """Generate a unique ID for an item based on its content and filename"""
    # Create a hash based on the item's key fields to identify unique decisions
    identifier_fields = ['رقم القرار', 'تاريخ القرار'] if 'رقم القرار' in item else []
    identifier_parts = [str(item.get(field, '')) for field in identifier_fields if field in item]
    identifier_parts.append(filename)
    
    identifier_string = '|'.join(identifier_parts)
    return hashlib.md5(identifier_string.encode('utf-8')).hexdigest()[:12]

def load_chunks(folder_path, text_fields, folder_name=""):
    """
    Load JSON files from the folder and create chunks with metadata.
    text_fields: list of keys to concatenate for the embedding text.
    """
    print(f"Loading chunks from {folder_path}")
    chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {filename}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item_idx, item in enumerate(data):
                        unique_id = generate_unique_id(item, filename)
                        text_parts = [str(item.get(field, '')) for field in text_fields if field in item]
                        text = ' '.join(text_parts).strip()
                        if text:
                            # Split large texts
                            text_chunks = split_text(text)
                            for chunk_idx, text_chunk in enumerate(text_chunks):
                                chunks.append({
                                    'text': text_chunk,  # This is what gets embedded
                                    'metadata': item.copy(),  # Complete original data
                                    'filename': filename,
                                    'unique_id': unique_id,
                                    'embedding_source': 'combined_fields',
                                    'chunk_index': chunk_idx,
                                    'total_chunks': len(text_chunks)
                                })
                else:
                    unique_id = generate_unique_id(data, filename)
                    text_parts = [str(data.get(field, '')) for field in text_fields if field in data]
                    text = ' '.join(text_parts).strip()
                    if text:
                        # Split large texts
                        text_chunks = split_text(text)
                        for chunk_idx, text_chunk in enumerate(text_chunks):
                            chunks.append({
                                'text': text_chunk,  # This is what gets embedded
                                'metadata': data.copy(),  # Complete original data
                                'filename': filename,
                                'unique_id': unique_id,
                                'embedding_source': 'combined_fields',
                                'chunk_index': chunk_idx,
                                'total_chunks': len(text_chunks)
                            })
    print(f"Loaded {len(chunks)} chunks from {folder_name}.")
    return chunks

def load_decision_chunks(folder_path, decision_fields, folder_name=""):
    """
    Load JSON files from the decisions folder and create separate chunks for each decision field.
    Each field gets its own embedding, but each chunk contains ALL the original data.
    
    decision_fields: list of keys to create separate embeddings for.
    """
    print(f"Loading decision chunks from {folder_path}")
    chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {filename}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item_idx, item in enumerate(data):
                        unique_id = generate_unique_id(item, filename)
                        # Create separate chunks for each decision field
                        for field in decision_fields:
                            if field in item:
                                # The text used for embedding is ONLY this field
                                embedding_text = str(item.get(field, '')).strip()
                                if embedding_text:
                                    # Split large texts
                                    text_chunks = split_text(embedding_text)
                                    for chunk_idx, text_chunk in enumerate(text_chunks):
                                        chunks.append({
                                            'text': text_chunk,  # Only this field's text for embedding
                                            'metadata': item.copy(),  # Complete original data with ALL fields
                                            'filename': filename,
                                            'unique_id': unique_id,
                                            'embedding_source': field,  # Which field was used for embedding
                                            'chunk_index': chunk_idx,
                                            'total_chunks': len(text_chunks),
                                            # Unique identifier for this specific field+chunk combination
                                            'field_chunk_id': f"{unique_id}_{field}_{chunk_idx}"
                                        })
                else:
                    unique_id = generate_unique_id(data, filename)
                    # Create separate chunks for each decision field
                    for field in decision_fields:
                        if field in data:
                            # The text used for embedding is ONLY this field
                            embedding_text = str(data.get(field, '')).strip()
                            if embedding_text:
                                # Split large texts
                                text_chunks = split_text(embedding_text)
                                for chunk_idx, text_chunk in enumerate(text_chunks):
                                    chunks.append({
                                        'text': text_chunk,  # Only this field's text for embedding
                                        'metadata': data.copy(),  # Complete original data with ALL fields
                                        'filename': filename,
                                        'unique_id': unique_id,
                                        'embedding_source': field,  # Which field was used for embedding
                                        'chunk_index': chunk_idx,
                                        'total_chunks': len(text_chunks),
                                        # Unique identifier for this specific field+chunk combination
                                        'field_chunk_id': f"{unique_id}_{field}_{chunk_idx}"
                                    })
    print(f"Loaded {len(chunks)} decision chunks from {folder_name}.")
    return chunks

def create_and_save_faiss(chunks, embeddings, index_filename, chunks_filename):
    """
    Create FAISS index and save it along with chunks.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    faiss.write_index(index, index_filename)
    
    with open(chunks_filename, 'wb') as f:
        pickle.dump(chunks, f)
    
    print(f"Saved FAISS index to {index_filename} and chunks to {chunks_filename}.")

def main():
    # Get API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("API key is required.")
        return

    # Folder paths
    # guides_folder = r"C:\Users\AReda\Desktop\finbot pro\final-data\guides"
    decisions_folder = r"C:\Users\AReda\Desktop\finbot pro\tt"

    # Load and embed guides (unchanged)
    # print("Loading and embedding guides chunks...")
    # guides_fields = ['filename', 'title', 'clean_content']
    # guides_chunks = load_chunks(guides_folder, guides_fields, "guides")
    
    # if guides_chunks:
    #     guides_texts = [chunk['text'] for chunk in guides_chunks]
    #     print(f"Embedding {len(guides_texts)} guide chunks...")
    #     guides_embeddings = np.vstack([get_embedding(text, api_key) for text in guides_texts])
    #     create_and_save_faiss(guides_chunks, guides_embeddings, 'guides.faiss', 'guides_chunks.pkl')
    # else:
    #     print("No guides chunks to process.")

    # Load and embed decisions (with separate embeddings for each field)
    # Only embedding the 3 main content fields
    print("Loading and embedding decisions chunks...")
    decisions_fields = [
        'اسباب القرار',
        'البنود محل الاعتراض',
        'منطوق القرار'
    ]
    decisions_chunks = load_decision_chunks(decisions_folder, decisions_fields, "decisions")
    
    if decisions_chunks:
        decisions_texts = [chunk['text'] for chunk in decisions_chunks]
        print(f"Embedding {len(decisions_texts)} decision chunks...")
        decisions_embeddings = np.vstack([get_embedding(text, api_key) for text in decisions_texts])
        create_and_save_faiss(decisions_chunks, decisions_embeddings, 'decisions.faiss', 'decisions_chunks.pkl')
    else:
        print("No decisions chunks to process.")

    print("Embedding process completed!")

if __name__ == "__main__":
    main()