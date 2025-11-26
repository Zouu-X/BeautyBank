# VGG similarity optimization with DDP
import argparse
import os
import torch
import torch.distributed as dist
import numpy as np
from pathlib import Path
from tqdm import tqdm

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, world_size, local_rank
    else:
        print("Not using distributed mode. Run with torchrun for DDP.")
        return 0, 1, 0

def load_and_search(feature_space_dir, query_path, batch_size=100, top_k=3):
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if rank == 0:
        print(f"Using {world_size} processes.")

    # Load query
    query_np = np.load(query_path)
    query = torch.from_numpy(query_np).to(device)
    
    # Ensure query is normalized
    # query = torch.nn.functional.normalize(query, p=2, dim=0) 

    all_files = sorted(Path(feature_space_dir).glob("*.npy"))
    if not all_files:
        if rank == 0:
            print("No embedding files found.")
        return

    # Partition files among ranks
    my_files = all_files[rank::world_size]
    num_files = len(my_files)
    
    if rank == 0:
        print(f"Total files: {len(all_files)}. Processing {num_files} files per rank (approx).")

    local_scores = []
    local_filenames = []
    
    # Process in batches
    # Only show progress bar on rank 0
    iterator = tqdm(range(0, num_files, batch_size), desc=f"Rank {rank}") if rank == 0 else range(0, num_files, batch_size)
    
    for i in iterator:
        batch_files = my_files[i : i + batch_size]
        batch_feats = []
        batch_names = []
        
        for f in batch_files:
            try:
                feat = np.load(f)
                batch_feats.append(feat)
                batch_names.append(f.name)
            except Exception as e:
                print(f"Rank {rank}: Error loading {f}: {e}")
                continue
        
        if not batch_feats:
            continue

        # Stack and move to GPU
        batch_tensor = torch.from_numpy(np.stack(batch_feats)).to(device)
        
        # Compute cosine similarity
        batch_scores_tensor = torch.mv(batch_tensor, query)
        
        local_scores.extend(batch_scores_tensor.cpu().tolist())
        local_filenames.extend(batch_names)

    # Find local top-k
    local_scores_np = np.array(local_scores)
    local_filenames_np = np.array(local_filenames)
    
    if len(local_scores_np) > 0:
        # Get top-k indices locally
        k = min(top_k, len(local_scores_np))
        top_k_local_indices = np.argsort(local_scores_np)[-k:][::-1]
        
        top_k_scores = local_scores_np[top_k_local_indices].tolist()
        top_k_names = local_filenames_np[top_k_local_indices].tolist()
        
        local_results = list(zip(top_k_names, top_k_scores))
    else:
        local_results = []

    # Gather results from all ranks
    all_results = [None for _ in range(world_size)]
    dist.all_gather_object(all_results, local_results)

    # Rank 0 merges and prints
    if rank == 0:
        # Flatten the list of lists
        flat_results = [item for sublist in all_results for item in sublist]
        
        # Sort by score descending
        flat_results.sort(key=lambda x: x[1], reverse=True)
        
        # Take global top-k
        global_top_k = flat_results[:top_k]
        
        print(f"\nTop-{top_k} matches (cosine similarity):")
        for r, (name, score) in enumerate(global_top_k, start=1):
            print(f"{r}. {name}: cosine={score:.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate cosine similarity for VGG embeddings with DDP.")
    parser.add_argument("--feature_space_dir", type=str, required=True, help="Directory containing .npy embedding files.")
    parser.add_argument("--query_path", type=str, required=True, help="Path to the query .npy file.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top matches to return.")
    
    args = parser.parse_args()

    load_and_search(args.feature_space_dir, args.query_path, args.batch_size, args.top_k)