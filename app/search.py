import os
import sys
import faiss
import torch
import numpy as np
from db import MediaDatabase
from transformers import CLIPModel, CLIPProcessor

# -----------------------------
# CONFIGURATION & PATHS
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "clip-vit-base-patch32")
FAISS_DB_DIR = os.path.join(SCRIPT_DIR, "faiss_db")

INDEX_FILE = os.path.join(FAISS_DB_DIR, "index.faiss")
SQL_DB_FILE = os.path.join(FAISS_DB_DIR, "localmind.db")

def perform_search(query, index, model, processor, db, top_k=5):
    """Executes a search query and returns the matching paths and confidence scores."""
    # Process text query
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embeds = model.get_text_features(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask']
        )
    
    # Extract and normalize the query vector
    query_vector = text_embeds.cpu().numpy().astype("float32")
    query_vector /= np.linalg.norm(query_vector, axis=1, keepdims=True)

    # Search FAISS index
    k = min(top_k, index.ntotal)
    if k == 0:
        return []

    distances, indices = index.search(query_vector, k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        # Look up path from SQLite database
        path = db.get_path_by_id(int(idx))
        if path:
            confidence = float(distances[0][rank])
            results.append((path, confidence))
            
    return results

def main():
    # 1. Validation checks
    if not os.path.exists(INDEX_FILE) or not os.path.exists(SQL_DB_FILE):
        print("Error: FAISS index or SQLite database not found!")
        print(f"Please run the indexing script first:\n  python index.py")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Downloading from Hugging Face...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print(f"Saving model locally to {MODEL_PATH}...")
        os.makedirs(MODEL_PATH, exist_ok=True)
        model.save_pretrained(MODEL_PATH)
        processor.save_pretrained(MODEL_PATH)
    else:
        print(f"Loading CLIP model from local path: {MODEL_PATH}")
        model = CLIPModel.from_pretrained(MODEL_PATH)
        processor = CLIPProcessor.from_pretrained(MODEL_PATH)

    print("Loading FAISS index & SQL database...")
    index = faiss.read_index(INDEX_FILE)
    db = MediaDatabase(SQL_DB_FILE)

    print(f"SimSearch Search Engine Ready. Total indexed images: {index.ntotal}")

    # Check if a query was provided as command line arguments
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nSearching for: '{query}'")
        try:
            results = perform_search(query, index, model, processor, db)
            if not results:
                print("No results found.")
            else:
                print("\nResults:")
                for rank, (path, confidence) in enumerate(results):
                    print(f"  {rank+1}. [Confidence: {confidence:.4f}] {path}")
        except Exception as e:
            print(f"Search failed: {e}")
            sys.exit(1)
    else:
        # Fallback to interactive search loop
        print("\nEntering interactive search mode. Type 'exit' to quit.")
        while True:
            try:
                query = input("\nSearch: ").strip()
                if not query:
                    continue
                if query.lower() in ("exit", "quit"):
                    print("Exiting search.")
                    break

                results = perform_search(query, index, model, processor, db)
                if not results:
                    print("No results found.")
                else:
                    print("\nResults:")
                    for rank, (path, confidence) in enumerate(results):
                        print(f"  {rank+1}. [Confidence: {confidence:.4f}] {path}")
            except KeyboardInterrupt:
                print("\nExiting search.")
                break
            except Exception as e:
                print(f"Search failed: {e}")

if __name__ == "__main__":
    main()
