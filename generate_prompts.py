import json
import pandas as pd
from kgtb.data.loader import DataLoader as KGTBFileDataLoader


def generate_prompts(location):
    print(f"Generating prompts for {location}...")

    with open(f"stru_ids_{location}.json", 'r') as f:
        stru_id_map = json.load(f)

    train_loader = KGTBFileDataLoader(loc=location, mode='train')
    val_loader = KGTBFileDataLoader(loc=location, mode='val')

    train_poi_df = train_loader.poi_metadata_df().rename(columns={'Pid': 'poi_id', 'Catname': 'category', 'Region': 'region', 'Latitude': 'latitude', 'Longitude': 'longitude'})
    val_poi_df = val_loader.poi_metadata_df().rename(columns={'Pid': 'poi_id', 'Catname': 'category', 'Region': 'region', 'Latitude': 'latitude', 'Longitude': 'longitude'})

    full_poi_df = pd.concat([train_poi_df, val_poi_df])

    data_df = pd.read_csv(f"kgtb/data/dataset/{location}/filtered_{location}_train.csv")

    poi_to_cat = pd.Series(full_poi_df.category.values, index=full_poi_df.poi_id).to_dict()
    poi_to_region = pd.Series(full_poi_df.region.values, index=full_poi_df.poi_id).to_dict()

    prompts = []

    trajectories = data_df.groupby('Uid')

    for user_id, trajectory_df in trajectories:
        trajectory_df = trajectory_df.sort_values(by='Local Time').reset_index(drop=True)

        poi_sequence = trajectory_df['Pid'].tolist()

        if len(poi_sequence) < 2:
            continue

        history_pois = poi_sequence[:-1]
        target_poi = poi_sequence[-1]

        user_stru_id_str = stru_id_map.get(str(user_id))

        if not user_stru_id_str:
            continue

        pref_pois_str_list = []
        unique_history_pois = pd.Series(history_pois).unique()
        for poi in unique_history_pois:
            poi_stru_id = stru_id_map.get(str(poi))
            cat_stru_id = stru_id_map.get(str(poi_to_cat.get(poi)))
            region_stru_id = stru_id_map.get(str(poi_to_region.get(poi)))
            if all(v is not None for v in [poi_stru_id, cat_stru_id, region_stru_id]):
                pref_pois_str_list.append(f"{cat_stru_id} {region_stru_id} {poi_stru_id}")

        pref_pois_str = ", ".join(pref_pois_str_list)

        # Get trajectory string
        traj_str_list = []
        for i, poi in enumerate(history_pois):
            poi_stru_id = stru_id_map.get(str(poi))
            cat_stru_id = stru_id_map.get(str(poi_to_cat.get(poi)))
            region_stru_id = stru_id_map.get(str(poi_to_region.get(poi)))
            local_time = trajectory_df.iloc[i]['Local Time']

            if all(v is not None for v in [poi_stru_id, cat_stru_id, region_stru_id]):
                traj_str_list.append(f"visiting {cat_stru_id} {region_stru_id} {poi_stru_id} at time {local_time}")

        traj_str = ", ".join(traj_str_list)

        poi_input_block = f"Please conduct a next POI recommendation. There is user {user_stru_id_str} and his preferable POIs: {pref_pois_str}. Here is his current trajectory: {traj_str}. Which POI will the user {user_stru_id_str} visit at time t{len(history_pois)+1}?"

        target_poi_stru_id = stru_id_map.get(str(target_poi))
        if target_poi_stru_id is not None:
            poi_output_block = f"POI {target_poi_stru_id}"
            prompts.append({"input": poi_input_block, "output": poi_output_block, "task": "poi_prediction"})

        cat_input_block = f"Please conduct a next category recommendation. There is user {user_stru_id_str} and his preferable POIs: {pref_pois_str}. Here is his current trajectory: {traj_str}. Which category will the user {user_stru_id_str} visit at time t{len(history_pois)+1}?"

        target_cat_stru_id = stru_id_map.get(str(poi_to_cat.get(target_poi)))
        if target_cat_stru_id is not None:
            cat_output_block = f"Category {target_cat_stru_id}"
            prompts.append({"input": cat_input_block, "output": cat_output_block, "task": "category_prediction"})

        region_input_block = f"Please conduct a next region recommendation. There is user {user_stru_id_str} and his preferable POIs: {pref_pois_str}. Here is his current trajectory: {traj_str}. Which region will the user {user_stru_id_str} visit at time t{len(history_pois)+1}?"

        target_region_stru_id = stru_id_map.get(str(poi_to_region.get(target_poi)))
        if target_region_stru_id is not None:
            region_output_block = f"Region {target_region_stru_id}"
            prompts.append({"input": region_input_block, "output": region_output_block, "task": "region_prediction"})

    prompts_df = pd.DataFrame(prompts)
    output_path = f"generated_prompts_{location}.jsonl"
    prompts_df.to_json(output_path, orient='records', lines=True)
    print(f"Generated {len(prompts)} prompts and saved to {output_path}")


if __name__ == "__main__":
    for location in ["CA", "NYC", "TKY"]:
        generate_prompts(location)
