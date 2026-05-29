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

def train_shared_model(graphs, lr, epoch_num, device, encoder, lambda_loss1, lambda_loss2, hidden_dim, sample_size):
    first_g = graphs[0].to(device)
    first_feats = first_g.ndata["attr"]
    first_neighbor_dict = build_neighbor_dict(first_g)
    neighbor_num_list = [len(v) for v in first_neighbor_dict.values()]
    in_dim = first_feats.shape[1]

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

    history = []
    for epoch in tqdm(range(epoch_num), desc="epochs"):
        model.train()
        total_loss = 0.0

        for g in tqdm(graphs, desc=f"epoch {epoch:03d}", leave=False):
            g = g.to(device)
            feats = g.ndata["attr"].to(device)
            neighbor_dict = build_neighbor_dict(g)
            loss, _ = model(g, feats, g.in_degrees(), neighbor_dict, device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(graphs)
        history.append(avg_loss)
        print(f"epoch {epoch:03d} loss={avg_loss:.6f}")

    model.eval()
    graph_embeddings = []
    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            feats = g.ndata["attr"].to(device)
            neighbor_dict = build_neighbor_dict(g)
            _, node_embeddings = model(g, feats, g.in_degrees(), neighbor_dict, device=device)
            graph_embeddings.append(graph_embedding_from_nodes(node_embeddings).cpu())

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
    parser.add_argument("--device", type=str, default="cpu")
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
