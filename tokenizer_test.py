"""
Main testing script for the KGTB model.

This script loads a pre-trained KGTB model and evaluates it on the test set.
"""
import json
import torch
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm

# Your custom modules
from kgtb.data.loader import DataLoader as KGTBFileDataLoader
from kgtb.model.kg import KGGraph
from kgtb.model.kgtb_model import KGTBModel


# --- Configuration ---
LOCATIONS = ["CA", "NYC", "TKY"]
BATCH_SIZE = 1024
DIST_THRESHOLD = 0.2

# --- Device Setup ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def convert_df_to_triples(df_visits, graph: KGGraph):
    """Helper function to convert validation/test dataframes to triple tensors."""
    triples = []
    r_vis = graph.relation_to_id['r_vis']
    for _, row in df_visits.iterrows():
        u = graph.entity_to_id.get(str(row['user_id']))
        v = graph.entity_to_id.get(str(row['poi_id']))
        if u is not None and v is not None:
            triples.append((u, r_vis, v))
    return torch.tensor(triples, dtype=torch.long)


def test_stru_ids(location, num_layers=3):
    """
    Tests the generated StruIDs JSON file for correctness.
    """
    print(f"--- Testing StruIDs for {location} ---")
    json_path = f"stru_ids_{location}.json"
    try:
        with open(json_path, 'r') as f:
            stru_id_map = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_path} not found.")
        return

    assert isinstance(stru_id_map, dict), "StruID JSON should contain a dictionary."
    print("Check 1/5: StruID file contains a dictionary... PASSED")

    if not stru_id_map:
        print("Warning: StruID map is empty.")
        return

    # Check a sample of items
    sample_key, sample_value = next(iter(stru_id_map.items()))

    assert isinstance(sample_key, str), "StruID keys (entity IDs) should be strings."
    print("Check 2/5: StruID keys are strings... PASSED")

    assert isinstance(sample_value, str), "StruID values should be strings."
    print("Check 3/5: StruID values are strings... PASSED")

    assert sample_value.startswith("stru_"), "StruID values should start with 'stru_'."
    print("Check 4/5: StruID values have the correct prefix... PASSED")

    num_parts = len(sample_value.split('_')) - 1
    assert num_parts == num_layers, f"Expected {num_layers} parts in StruID, but found {num_parts}."
    print(f"Check 5/5: StruID values have the correct number of parts ({num_layers})... PASSED")

    print(f"All StruID checks passed for {location}!")


@torch.no_grad()
def evaluate(model, triples_to_test, graph_structure, device):
    """
    Evaluates the model on a set of triples.
    """
    model.eval()
    test_dataset = TensorDataset(triples_to_test)
    test_loader = TorchDataLoader(test_dataset, batch_size=BATCH_SIZE)

    total_loss = 0
    for batch in tqdm(test_loader, desc="Evaluating"):
        positive_triples = batch[0].to(device)
        loss = model(graph_structure['edge_index'], graph_structure['edge_type'], positive_triples)
        total_loss += loss.item()

    return total_loss / len(test_loader)


for location in LOCATIONS:
    # --- 1. Data Loading (Train/Val for graph construction) ---
    print(f"Loading data for location: {location} ...")
    train_loader = KGTBFileDataLoader(loc=location, mode='train')
    val_loader = KGTBFileDataLoader(loc=location, mode='val')
    test_loader = KGTBFileDataLoader(loc=location, mode='test')

    train_visits_df = train_loader.get_visit_df().rename(columns={'Uid': 'user_id', 'Pid': 'poi_id'})
    train_poi_df = train_loader.poi_metadata_df().rename(columns={'Pid': 'poi_id', 'Catname': 'category', 'Region': 'region', 'Latitude': 'latitude', 'Longitude': 'longitude'})

    val_visits_df = val_loader.get_visit_df().rename(columns={'Uid': 'user_id', 'Pid': 'poi_id'})
    val_poi_df = val_loader.poi_metadata_df().rename(columns={'Pid': 'poi_id', 'Catname': 'category', 'Region': 'region', 'Latitude': 'latitude', 'Longitude': 'longitude'})

    test_visits_df = test_loader.get_visit_df().rename(columns={'Uid': 'user_id', 'Pid': 'poi_id'})

    graph = KGGraph()

    # --- 2. Graph and Mappings Construction (using train+val) ---
    print("Building entity mappings and graph structure...")
    full_visits_df = pd.concat([train_visits_df, val_visits_df])
    full_poi_df = pd.concat([train_poi_df, val_poi_df])

    graph.build_mappings(full_visits_df, full_poi_df)
    graph.construct_knowledge_graph(dist_thresh=DIST_THRESHOLD, user_visits_df=train_visits_df, poi_metadata_df=train_poi_df)

    # --- 3. Prepare Data for PyTorch ---
    graph_structure = graph.get_graph_tensors()
    graph_structure['edge_index'] = graph_structure['edge_index'].to(DEVICE)
    graph_structure['edge_type'] = graph_structure['edge_type'].to(DEVICE)

    test_triples = convert_df_to_triples(test_visits_df, graph)

    entity_type_map = graph.get_entity_type_map()
    for entity_type, indices in entity_type_map.items():
        entity_type_map[entity_type] = indices.to(DEVICE)

    num_nodes = graph.get_num_nodes()
    num_relations = graph.get_num_relations()

    # --- 4. Model Initialization and Loading ---
    model = KGTBModel(
        num_nodes=num_nodes,
        num_relations=num_relations,
        entity_type_map=entity_type_map
    ).to(DEVICE)

    model_path = f"kgtb_model_{location}_best.pt"
    print(f"Loading pre-trained model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # --- 5. Evaluation ---
    print(f"Evaluating model on the test set for {location}...")
    test_loss = evaluate(model, test_triples, graph_structure, DEVICE)
    print(f"Test Loss for {location}: {test_loss:.4f}")

    # --- 6. Test StruID Generation ---
    test_stru_ids(location)
