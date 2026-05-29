import os
import sys
import glob
import argparse
import random
import numpy as np
import torch
import dgl
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import GNNStructEncoder

FIXED_NODES = 200

def load_contact_map(path):
    adj = np.load(path)
    if adj.shape != (FIXED_NODES, FIXED_NODES):
        raise ValueError(f"Expected {FIXED_NODES}x{FIXED_NODES}, got {adj.shape}")
    src, dst = np.nonzero(adj > 0)
    g = dgl.graph((src, dst), num_nodes=FIXED_NODES)
    g = dgl.add_self_loop(g)
    g.ndata["attr"] = torch.eye(FIXED_NODES, dtype=torch.float32)
    return g

def build_neighbor_dict(g):
    in_nodes, out_nodes = g.edges()
    neighbor_dict = {}
    for u, v in zip(in_nodes.tolist(), out_nodes.tolist()):
        neighbor_dict.setdefault(u, []).append(v)
    return neighbor_dict

def graph_embedding_from_nodes(node_embeddings):
    return node_embeddings.mean(dim=0)

def train_shared_model(graphs, lr, epoch_num, device, encoder, lambda_loss1, lambda_loss2, hidden_dim, sample_size, batch_size=32):
    individual_neighbor_dicts = [build_neighbor_dict(g) for g in graphs]
    
    first_g = graphs[0]
    in_dim = first_g.ndata["attr"].shape[1]
    neighbor_num_list = [len(v) for v in individual_neighbor_dicts[0].values()]

    model = GNNStructEncoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        layer_num=2,
        sample_size=sample_size,
        device=device,
        neighbor_num_list=neighbor_num_list,
        GNN_name=encoder,
        lambda_loss1=lambda_loss1,
        lambda_loss2=lambda_loss2,
    ).to(device)

    degree_params = list(map(id, model.degree_decoder.parameters()))
    base_params = filter(lambda p: id(p) not in degree_params, model.parameters())
    optimizer = torch.optim.Adam(
        [{"params": base_params}, {"params": model.degree_decoder.parameters(), "lr": 1e-2}],
        lr=lr,
        weight_decay=3e-4,
    )

    # Prepare batches
    num_graphs = len(graphs)
    history = []
    
    for epoch in tqdm(range(epoch_num), desc="epochs"):
        model.train()
        total_loss = 0.0
        
        # Shuffle indices
        indices = list(range(num_graphs))
        random.shuffle(indices)
        
        pbar = tqdm(range(0, num_graphs, batch_size), desc=f"epoch {epoch:03d}", leave=False)
        for i in pbar:
            batch_idx = indices[i : i + batch_size]
            batch_graphs = [graphs[j] for j in batch_idx]
            
            # Batch graphs with DGL
            bg = dgl.batch(batch_graphs).to(device)
            feats = bg.ndata["attr"].to(device)
            degrees = bg.in_degrees().to(device)
            
            # Construct a merged neighbor_dict for the batch
            # dgl.batch shifts node IDs. We need to mirror that.
            merged_neighbor_dict = {}
            node_offset = 0
            for j in batch_idx:
                g_neighbor_dict = individual_neighbor_dicts[j]
                num_nodes = graphs[j].num_nodes()
                for u, neighbors in g_neighbor_dict.items():
                    merged_neighbor_dict[u + node_offset] = [v + node_offset for v in neighbors]
                node_offset += num_nodes
                
            loss, _ = model(bg, feats, degrees, merged_neighbor_dict, device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_idx)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_graphs
        history.append(avg_loss)
        print(f"epoch {epoch:03d} loss={avg_loss:.6f}")

    model.eval()
    graph_embeddings = []
    with torch.no_grad():
        for i in range(0, num_graphs, batch_size):
            batch_graphs = graphs[i : i + batch_size]
            bg = dgl.batch(batch_graphs).to(device)
            feats = bg.ndata["attr"].to(device)
            degrees = bg.in_degrees().to(device)
            
            merged_neighbor_dict = {}
            node_offset = 0
            for j in range(i, min(i + batch_size, num_graphs)):
                g_neighbor_dict = individual_neighbor_dicts[j]
                num_nodes = graphs[j].num_nodes()
                for u, neighbors in g_neighbor_dict.items():
                    merged_neighbor_dict[u + node_offset] = [v + node_offset for v in neighbors]
                node_offset += num_nodes

            _, node_embeddings = model(bg, feats, degrees, merged_neighbor_dict, device=device)
            
            # Split node embeddings back to individual graphs and pool
            node_offset = 0
            for g_in_batch in batch_graphs:
                num_nodes = g_in_batch.num_nodes()
                g_node_embeddings = node_embeddings[node_offset : node_offset + num_nodes]
                graph_embeddings.append(graph_embedding_from_nodes(g_node_embeddings).cpu())
                node_offset += num_nodes

    return model, torch.stack(graph_embeddings, dim=0), history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="cm-dataset")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epoch_num", type=int, default=50)
    parser.add_argument("--lambda_loss1", type=float, default=1e-2)
    parser.add_argument("--lambda_loss2", type=float, default=1.0)
    parser.add_argument("--sample_size", type=int, default=5)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--encoder", type=str, default="GCN", choices=["GCN", "GIN", "SAGE"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    npy_files = sorted(glob.glob(os.path.join(args.data_dir, "*.npy")))
    if not npy_files:
        raise RuntimeError(f"No .npy files found in {args.data_dir}")

    graphs = [load_contact_map(p) for p in npy_files]
    names = [os.path.splitext(os.path.basename(p))[0] for p in npy_files]

    os.makedirs("output", exist_ok=True)

    model, graph_embeddings, history = train_shared_model(
        graphs=graphs,
        lr=args.lr,
        epoch_num=args.epoch_num,
        device=device,
        encoder=args.encoder,
        lambda_loss1=args.lambda_loss1,
        lambda_loss2=args.lambda_loss2,
        hidden_dim=args.dimension,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
    )

    torch.save(model.state_dict(), "output/shared_model.pt")
    torch.save(graph_embeddings, "output/graph_embeddings.pt")

    import pandas as pd
    emb_df = pd.DataFrame(graph_embeddings.numpy())
    emb_df.insert(0, "name", names)
    emb_df.to_csv("output/graph_embeddings.csv", index=False)

    loss_df = pd.DataFrame({"epoch": list(range(len(history))), "loss": history})
    loss_df.to_csv("output/train_loss.csv", index=False)

    print("Saved output/shared_model.pt")
    print("Saved output/graph_embeddings.pt")
    print("Saved output/graph_embeddings.csv")
    print("Saved output/train_loss.csv")

if __name__ == "__main__":
    main()
