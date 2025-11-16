"""
Main training script for the KGTB model.

This script orchestrates the entire training, validation, and final inference process.
"""
import json
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm

# Your custom modules
from kgtb.data.loader import DataLoader as KGTBFileDataLoader
from kgtb.model.kg import KGGraph
from kgtb.model.kgtb_model import KGTBModel


# --- Configuration ---
LOCATIONS = ["CA", "NYC", "TKY"]
LEARNING_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 1024
DIST_THRESHOLD = 0.2

# --- Device Setup ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


def convert_df_to_triples(df_visits, df_poi, graph: KGGraph):
    """Helper function to convert validation/test dataframes to triple tensors."""
    triples = []
    # This logic should mirror the triple creation in your KGGraph class
    # to ensure consistency.
    r_vis = graph.relation_to_id['r_vis']
    for _, row in df_visits.iterrows():
        # Use .get() to avoid errors if an entity was not in the training set mappings
        u = graph.entity_to_id.get(str(row['user_id']))
        v = graph.entity_to_id.get(str(row['poi_id']))
        if u is not None and v is not None:
            triples.append((u, r_vis, v))
    # ... you would add similar logic for cat, loc, adj if they are in val/test sets
    return torch.tensor(triples, dtype=torch.long)


@torch.no_grad()
def evaluate(model, triples_to_test, graph_structure, device):
    """
    Evaluates the model on a set of triples using MRR and Hits@K metrics.
    (This is a simplified version for demonstration)
    """
    model.eval()
    # A full evaluation is computationally expensive. For this script, we will
    # just calculate the loss on the test set as a proxy for performance.
    # A real implementation would calculate MRR/Hits@K as discussed previously.

    test_dataset = TensorDataset(triples_to_test)
    test_loader = TorchDataLoader(test_dataset, batch_size=BATCH_SIZE)

    total_loss = 0
    for batch in tqdm(test_loader, desc="Evaluating"):
        positive_triples = batch[0].to(device)
        loss = model(graph_structure['edge_index'], graph_structure['edge_type'], positive_triples)
        total_loss += loss.item()

    return total_loss / len(test_loader)


for location in LOCATIONS:
    # --- 1. Data Loading ---
    print(f"Loading data files for all splits in location: {location} ...")
    train_loader = KGTBFileDataLoader(loc=location, mode='train')
    val_loader = KGTBFileDataLoader(loc=location, mode='val')
    # test_loader = KGTBFileDataLoader(loc=LOCATION, mode='test') # Uncomment for final testing

    train_visits_df = train_loader.get_visit_df().rename(columns={'Uid': 'user_id', 'Pid': 'poi_id'})
    train_poi_df = train_loader.poi_metadata_df().rename(columns={'Pid': 'poi_id', 'Catname': 'category', 'Region': 'region', 'Latitude': 'latitude', 'Longitude': 'longitude'})

    val_visits_df = val_loader.get_visit_df().rename(columns={'Uid': 'user_id', 'Pid': 'poi_id'})
    val_poi_df = val_loader.poi_metadata_df().rename(columns={'Pid': 'poi_id', 'Catname': 'category', 'Region': 'region', 'Latitude': 'latitude', 'Longitude': 'longitude'})

    graph = KGGraph()

    # --- 2. Graph and Mappings Construction ---
    print("Building consistent entity mappings from all data...")
    full_visits_df = pd.concat([train_visits_df, val_visits_df])
    full_poi_df = pd.concat([train_poi_df, val_poi_df])

    # Initialize KGGraph with the complete data to build full mappings

    graph.build_mappings(full_visits_df, full_poi_df)

    print("Constructing graph structure from TRAINING data only...")
    # Now, construct the graph structure using ONLY the training dataframes
    graph.construct_knowledge_graph(dist_thresh=DIST_THRESHOLD, user_visits_df=train_visits_df, poi_metadata_df=train_poi_df)

    # --- 3. Prepare Data for PyTorch ---
    graph_structure = graph.get_graph_tensors()
    graph_structure['edge_index'] = graph_structure['edge_index'].to(DEVICE)
    graph_structure['edge_type'] = graph_structure['edge_type'].to(DEVICE)

    train_triples_np = np.array(graph.triples)
    train_triples = torch.tensor(train_triples_np, dtype=torch.long)

    # Convert validation dataframes to triples using the complete mappings
    val_triples = convert_df_to_triples(val_visits_df, val_poi_df, graph)

    entity_type_map = graph.get_entity_type_map()
    for entity_type, indices in entity_type_map.items():
        entity_type_map[entity_type] = indices.to(DEVICE)

    num_nodes = graph.get_num_nodes()
    num_relations = graph.get_num_relations()

    train_dataset = TensorDataset(train_triples)
    train_torch_loader = TorchDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 4. Model Initialization ---
    # The embedding dimension of the KGTB model is specified as 64 in the paper.
    # This is the dimension of the code vectors, not the final LLM embedding space.
    KGTB_EMBEDDING_DIM = 64

    model = KGTBModel(
        num_nodes=num_nodes,
        num_relations=num_relations,
        entity_type_map=entity_type_map,
        dim=KGTB_EMBEDDING_DIM
    ).to(DEVICE)

    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # --- 5. Training & Validation Loop ---
    best_val_loss = float('inf')

    print(f"Starting training for location: {location}")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_torch_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            positive_triples = batch[0].to(DEVICE)

            optimizer.zero_grad()
            loss = model(graph_structure['edge_index'], graph_structure['edge_type'], positive_triples)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_torch_loader)

        # Validation phase
        avg_val_loss = evaluate(model, val_triples, graph_structure, DEVICE)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("New best validation loss! Saving model checkpoint...")
            torch.save(model.state_dict(), f"kgtb_model_{location}_best.pt")

        # --- Save stru ids (tensor + json) for this epoch ---
        # Ensure output directories exist
        base_out_dir = os.path.join(".", "quantizer_output", location)
        json_out_dir = os.path.join(base_out_dir, "stru_ids")
        vec_out_dir = os.path.join(base_out_dir, "vectors")
        os.makedirs(json_out_dir, exist_ok=True)
        os.makedirs(vec_out_dir, exist_ok=True)

        # Encode current model state into stru ids
        with torch.no_grad():
            epoch_stru_tensor = model.encode(
                graph_structure['edge_index'],
                graph_structure['edge_type']
            )

        # Save tensor (CPU) and JSON mapping
        epoch_idx = epoch + 1  # 1-based epoch numbering in filenames
        vector_output_path = os.path.join(vec_out_dir, f"stru_id_epoch_{epoch_idx}.pt")
        torch.save(epoch_stru_tensor.cpu(), vector_output_path)

        # Build string tokens and entity mapping
        stru_ids_epoch_list = [f"stru_{'_'.join(map(str, ids.tolist()))}" for ids in epoch_stru_tensor]
        id_to_entity = {v: k for k, v in graph.entity_to_id.items()}
        stru_id_epoch_map = {
            id_to_entity[i]: stru_ids_epoch_list[i]
            for i in range(len(stru_ids_epoch_list))
        }

        json_output_path = os.path.join(json_out_dir, f"stru_id_epoch_{epoch_idx}.json")
        with open(json_output_path, 'w') as jf:
            json.dump(stru_id_epoch_map, jf, indent=4)
        print(f"Saved epoch {epoch_idx} stru ids: tensor -> {vector_output_path}, json -> {json_output_path}")

    # --- 6. Final Inference ---
    print("Training complete. Loading best model for final encoding.")
    model.load_state_dict(torch.load(f"kgtb_model_{location}_best.pt"))

    with torch.no_grad():
        final_stru_ids_tensor = model.encode(
            graph_structure['edge_index'],
            graph_structure['edge_type']
        )

    print(f"Generated StruIDs tensor with shape: {final_stru_ids_tensor.shape}")

    # Save the final stru ids tensor for downstream evaluation or inspection.
    # Save a CPU copy to avoid GPU-device pickle issues.
    tensor_save_path = f"final_stru_ids_tensor_{location}.pt"
    torch.save(final_stru_ids_tensor.cpu(), tensor_save_path)
    print(f"Saved final_stru_ids_tensor to {tensor_save_path}")

    # Convert tensor to a list of string-formatted StruIDs
    stru_ids_list = [f"stru_{'_'.join(map(str, ids.tolist()))}" for ids in final_stru_ids_tensor]

    # Create a mapping from the original entity ID to the new string StruID
    # We need to invert the entity_to_id mapping to get id_to_entity
    id_to_entity = {v: k for k, v in graph.entity_to_id.items()}

    stru_id_map = {
        id_to_entity[i]: stru_ids_list[i]
        for i in range(len(stru_ids_list))
    }

    # Save the mapping to a JSON file
    output_path = f"stru_ids_{location}.json"
    with open(output_path, 'w') as f:
        json.dump(stru_id_map, f, indent=4)

    print(f"Final StruID mappings saved to {output_path}")
