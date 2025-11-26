# VGG similarity optimization
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

def load_and_search(feature_space_dir, query_path, batch_size=100, top_k=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load query
    query_np = np.load(query_path)
    query = torch.from_numpy(query_np).to(device)
    
    # Ensure query is normalized (it should be from vgg_face_similarity.py, but good to be safe)
    # query = torch.nn.functional.normalize(query, p=2, dim=0) 

    files = sorted(Path(feature_space_dir).glob("*.npy"))
    if not files:
        print("No embedding files found.")
        return

    scores_list = []
    indices_list = []
    
    # Process in batches
    num_files = len(files)
    print(f"Processing {num_files} files in batches of {batch_size}...")
    
    for i in tqdm(range(0, num_files, batch_size)):
        batch_files = files[i : i + batch_size]
        batch_feats = []
        valid_indices = []
        
        for j, f in enumerate(batch_files):
            try:
                feat = np.load(f)
                batch_feats.append(feat)
                valid_indices.append(i + j)
            except Exception as e:
                print(f"Error loading {f}: {e}")
                continue
        
        if not batch_feats:
            continue

        # Stack and move to GPU
        batch_tensor = torch.from_numpy(np.stack(batch_feats)).to(device)
        
        # Compute cosine similarity
        # batch_tensor shape: (B, D), query shape: (D,)
        # result shape: (B,)
        batch_scores = torch.mv(batch_tensor, query)
        
        scores_list.extend(batch_scores.cpu().tolist())
        indices_list.extend(valid_indices)

    # Find top-k
    scores_np = np.array(scores_list)
    indices_np = np.array(indices_list)
    
    if len(scores_np) == 0:
        print("No valid scores computed.")
        return

    # Get top-k indices in the scores array
    # Note: argsort sorts in ascending order, so we take the last k
    top_k_local_indices = np.argsort(scores_np)[-top_k:][::-1]
    
    print(f"\nTop-{top_k} matches (cosine similarity):")
    for rank, local_idx in enumerate(top_k_local_indices, start=1):
        global_idx = indices_np[local_idx]
        score = scores_np[local_idx]
        file_name = files[global_idx].name
        print(f"{rank}. {file_name}: cosine={score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate cosine similarity for VGG embeddings.")
    parser.add_argument("--feature_space_dir", type=str, required=True, help="Directory containing .npy embedding files.")
    parser.add_argument("--query_path", type=str, required=True, help="Path to the query .npy file.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top matches to return.")
    
    args = parser.parse_args()

    load_and_search(args.feature_space_dir, args.query_path, args.batch_size, args.top_k)