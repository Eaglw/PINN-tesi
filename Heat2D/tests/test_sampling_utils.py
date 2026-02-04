import torch
import pytest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from sampling_utils import generate_internal_points, generate_grid_points, filter_and_refill

def test_generate_internal_points_margin():
    Lx, Ly = 1.0, 1.0
    margin = 0.1
    num_points = 1000
    points = generate_internal_points(num_points, Lx, Ly, margin)
    
    assert points.shape == (num_points, 2)
    assert torch.all(points[:, 0] >= margin)
    assert torch.all(points[:, 0] <= Lx - margin)
    assert torch.all(points[:, 1] >= margin)
    assert torch.all(points[:, 1] <= Ly - margin)

def test_generate_grid_points_margin():
    Lx, Ly = 1.0, 1.0
    margin = 0.1
    Nx, Ny = 10, 10
    points = generate_grid_points(Nx, Ny, Lx, Ly, margin)
    
    assert points.shape == (Nx * Ny, 2)
    assert torch.all(points[:, 0] >= margin - 1e-10) # tolerance for float
    assert torch.all(points[:, 0] <= Lx - margin + 1e-10)
    assert torch.all(points[:, 1] >= margin - 1e-10)
    assert torch.all(points[:, 1] <= Ly - margin + 1e-10)

def test_filter_and_refill_disjointness():
    primary_set = torch.tensor([[0.5, 0.5], [0.1, 0.1]])
    d_min = 0.2
    target_count = 5
    
    def generator(n):
        # Always generates points that would be filtered if we are not careful
        # But here we just use random
        return torch.rand((n, 2))
    
    final_set = filter_and_refill(primary_set, generator, target_count, d_min)
    
    assert final_set.shape == (target_count, 2)
    
    # Check distances to primary_set
    dists = torch.cdist(final_set, primary_set)
    assert torch.all(dists >= d_min)
    
    # Check self-disjointness
    if target_count > 1:
        for i in range(target_count):
            other_points = torch.cat([final_set[:i], final_set[i+1:]])
            dist_to_others = torch.norm(final_set[i] - other_points, dim=1)
            assert torch.all(dist_to_others >= d_min)

def test_filter_and_refill_target_count():
    # If many points are filtered, it should still reach target_count
    primary_set = torch.rand((100, 2))
    d_min = 0.05
    target_count = 50
    
    def generator(n):
        return torch.rand((n, 2))
    
    final_set = filter_and_refill(primary_set, generator, target_count, d_min)
    assert final_set.shape[0] == target_count
