import torch
import numpy as np
from sklearn.neighbors import BallTree


class KGGraph:
    def __init__(self):
        self.triples = []
        self.entity_to_id = {}
        self.relation_to_id = {}
        self.entity_type_map = {}

        relations = ['r_vis', 'r_adj', 'r_cat', 'r_loc', 'r_vis_inv', 'r_adj_inv', 'r_cat_inv', 'r_loc_inv']
        self.relation_to_id = {rel: i for i, rel in enumerate(relations)}

    def build_mappings(self, all_user_visits_df, all_poi_metadata_df):
        print("Building entity mappings and type map from all data splits...")
        all_users = all_user_visits_df['user_id'].unique()
        all_pois = all_poi_metadata_df['poi_id'].unique()
        all_categories = all_poi_metadata_df['category'].unique()
        all_regions = all_poi_metadata_df['region'].unique()

        all_entities = np.concatenate([
            all_users.astype(str),
            all_pois.astype(str),
            all_categories.astype(str),
            all_regions.astype(str)
        ])

        self.entity_to_id = {entity: i for i, entity in enumerate(np.unique(all_entities))}

        self.entity_type_map = {
            'user': torch.tensor([self.entity_to_id[str(e)] for e in all_users], dtype=torch.long),
            'poi': torch.tensor([self.entity_to_id[str(e)] for e in all_pois], dtype=torch.long),
            'category': torch.tensor([self.entity_to_id[str(e)] for e in all_categories], dtype=torch.long),
            'region': torch.tensor([self.entity_to_id[str(e)] for e in all_regions], dtype=torch.long)
        }

    def construct_knowledge_graph(self, user_visits_df, poi_metadata_df, dist_thresh):
        print("Constructing 'visit' relations...")
        for _, row in user_visits_df.iterrows():
            u = self.entity_to_id[str(row['user_id'])]
            v = self.entity_to_id[str(row['poi_id'])]
            r = self.relation_to_id['r_vis']
            self.triples.append((u, r, v))

        print("Constructing 'adjacent' relations...")
        poi_df = poi_metadata_df.set_index('poi_id')
        poi_df = poi_df[~poi_df.index.duplicated(keep='first')]

        coords = np.radians(poi_df[['latitude', 'longitude']].values)
        tree = BallTree(coords, metric='haversine')
        dist_thresh_rad = dist_thresh / 6371.0
        neighbors_indices = tree.query_radius(coords, r=dist_thresh_rad)

        for i, poi1_idx in enumerate(poi_df.index):
            for j in neighbors_indices[i]:
                if i == j:
                    continue
                poi2_idx = poi_df.index[j]
                if poi1_idx < poi2_idx:
                    v1 = self.entity_to_id[str(poi1_idx)]
                    v2 = self.entity_to_id[str(poi2_idx)]
                    r = self.relation_to_id['r_adj']
                    self.triples.append((v1, r, v2))
                    r = self.relation_to_id['r_adj_inv']
                    self.triples.append((v2, r, v1))

        print("Constructing 'categorized' relations...")
        for _, row in poi_metadata_df.iterrows():
            v = self.entity_to_id[str(row['poi_id'])]
            c = self.entity_to_id[str(row['category'])]
            r = self.relation_to_id['r_cat']
            self.triples.append((v, r, c))
            r = self.relation_to_id['r_cat_inv']
            self.triples.append((c, r, v))

        print("Constructing 'located' relations...")
        for _, row in poi_metadata_df.iterrows():
            v = self.entity_to_id[str(row['poi_id'])]
            o = self.entity_to_id[str(row['region'])]
            r = self.relation_to_id['r_loc']
            self.triples.append((v, r, o))
            r = self.relation_to_id['r_loc_inv']
            self.triples.append((o, r, v))

        print(f"Knowledge Graph constructed with {len(self.triples)} triples.")

    def get_graph_tensors(self):
        graph_triples = np.array(self.triples)
        edge_index = torch.tensor(np.array(graph_triples[:, [0, 2]].T), dtype=torch.long)
        edge_type = torch.tensor(np.array(graph_triples[:, 1]), dtype=torch.long)
        return {'edge_index': edge_index, 'edge_type': edge_type}

    def get_num_nodes(self):
        return len(self.entity_to_id)

    def get_num_relations(self):
        return len(self.relation_to_id)

    def get_entity_type_map(self):
        return self.entity_type_map
