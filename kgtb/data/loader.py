import pandas as pd


class DataLoader:
    def __init__(self, loc, mode):
        self.df = pd.read_csv(f"kgtb/data/dataset/{loc}/filtered_{loc}_{mode}.csv")

    def get_df(self):
        return self.df

    def get_visit_df(self):
        """:return: A DataFrame with columns ['Uid', 'Pid']."""
        return self.df[['Uid', 'Pid']].drop_duplicates().reset_index(drop=True)

    def poi_metadata_df(self):
        """:return: A DataFrame with columns ['Pid', 'Catname', 'Region', 'Latitude', 'Longitude']."""
        return self.df[['Pid', 'Catname', 'Region', 'Latitude', 'Longitude']].drop_duplicates().reset_index(drop=True)
