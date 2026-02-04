import torch

def generate_internal_points(num_points, Lx=1.0, Ly=1.0, margin=1e-5, device='cpu', dtype=torch.float64):
    """
    Generates random internal points within a safety margin from the boundaries.
    """
    xy = torch.rand((num_points, 2), device=device, dtype=dtype)
    # Scale and shift: [0, 1] -> [margin, Lx - margin]
    xy[:, 0] = xy[:, 0] * (Lx - 2 * margin) + margin
    xy[:, 1] = xy[:, 1] * (Ly - 2 * margin) + margin
    return xy

def generate_grid_points(Nx, Ny, Lx=1.0, Ly=1.0, margin=1e-5, device='cpu', dtype=torch.float64):
    """
    Generates internal grid points shifted away from the boundaries by a safety margin.
    """
    x = torch.linspace(margin, Lx - margin, Nx, device=device, dtype=dtype)
    y = torch.linspace(margin, Ly - margin, Ny, device=device, dtype=dtype)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    return torch.stack([X.flatten(), Y.flatten()], dim=1)

def filter_and_refill(primary_set, secondary_set_generator, target_count, d_min=1e-4):
    """
    Ensures disjointness between a primary set and a secondary set.
    Iteratively generates more points using secondary_set_generator until target_count is reached.
    
    Args:
        primary_set: Tensor of points that must be avoided.
        secondary_set_generator: Function that takes (num_to_generate) and returns a Tensor of points.
        target_count: Desired number of points in the resulting set.
        d_min: Minimum Euclidean distance.
    """
    device = primary_set.device
    dtype = primary_set.dtype
    final_set = torch.empty((0, 2), dtype=dtype, device=device)
    
    max_iters = 100
    iters = 0
    
    while final_set.shape[0] < target_count and iters < max_iters:
        needed = target_count - final_set.shape[0]
        # Generate some extra points to account for potential filtering
        candidates = secondary_set_generator(needed * 2) 
        
        # 1. Filter against Primary Set
        if primary_set.shape[0] > 0:
            dists_to_primary = torch.cdist(candidates, primary_set)
            min_dists_to_primary = torch.min(dists_to_primary, dim=1)[0]
            candidates = candidates[min_dists_to_primary >= d_min]
        
        # 2. Filter against already accumulated Final Set
        if final_set.shape[0] > 0 and candidates.shape[0] > 0:
            dists_to_final = torch.cdist(candidates, final_set)
            min_dists_to_final = torch.min(dists_to_final, dim=1)[0]
            candidates = candidates[min_dists_to_final >= d_min]
            
        if candidates.shape[0] == 0:
            iters += 1
            continue
            
        # 3. Greedy selection within the candidate batch to ensure self-disjointness
        # (This is O(N^2) for the batch, but batch size is small)
        valid_indices = []
        for i in range(candidates.shape[0]):
            if len(valid_indices) + final_set.shape[0] >= target_count:
                break
                
            cand = candidates[i].unsqueeze(0)
            
            # Check against currently selected candidates in this batch
            if len(valid_indices) > 0:
                current_batch_selected = candidates[valid_indices]
                dists = torch.norm(current_batch_selected - cand, dim=1)
                if torch.any(dists < d_min):
                    continue
            
            valid_indices.append(i)
            
        if len(valid_indices) > 0:
            to_add = candidates[valid_indices]
            final_set = torch.cat([final_set, to_add], dim=0)
            
        iters += 1
        
    if final_set.shape[0] < target_count:
        print(f"Warning: Could only generate {final_set.shape[0]}/{target_count} points after {max_iters} iterations.")
        
    return final_set

def check_overlaps(points, threshold=1e-7, label="Points"):
    """
    Checks if there are any points too close to each other in the given set.
    """
    if points.shape[0] < 2:
        return True
        
    # Compute pairwise distances
    dists = torch.cdist(points, points)
    
    # Fill diagonal with a value larger than threshold to ignore self-distance
    dists.fill_diagonal_(threshold * 10 + 1.0)
    
    min_dist = torch.min(dists).item()
    
    if min_dist < threshold:
        print(f"⚠️  [{label}] Overlap detected! Min distance: {min_dist:.2e}")
        return False
    else:
        print(f"✅ [{label}] No overlaps. Min distance: {min_dist:.2e}")
        return True
