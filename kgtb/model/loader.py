from kg import KGGraph
import torch
from typing import Tuple, Dict, Any
from kgtb.data.loader import DataLoader


def load_graph_data(loc: str, mode: str, device: torch.device) -> Tuple[Dict, Any, Any, Any, Any, Any]:
    dl = DataLoader(loc=loc, mode=mode)

    user_visits_df = dl.get_visit_df().rename(columns={'Uid': 'user_id', 'Pid': 'poi_id'})
    poi_metadata_df = dl.poi_metadata_df().rename(columns={
        'Pid': 'poi_id',
        'Catname': 'category',
        'Region': 'region',
        'Latitude': 'latitude',
        'Longitude': 'longitude'
    })

    graph = KGGraph(
        dist_thresh=0.2,  # 200 meters
        user_visits_df=user_visits_df,
        poi_metadata_df=poi_metadata_df
    )

    graph.build_mappings(user_visits_df, poi_metadata_df)
    graph.construct_knowledge_graph(user_visits_df, poi_metadata_df, dist_thresh=0.2)

    graph_data = {
        "tensors": graph.get_graph_tensros(device=device),
        "num_nodes": graph.get_num_nodes(),
        "num_relations": graph.get_num_relations(),
        "entity_type_map": graph.get_entity_type_map(device=device)
    }

    return graph_data
