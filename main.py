import yaml
from sentence_transformers import SentenceTransformer

from src import Ranking, Embeddings, load_dataset

with open("config.yaml", "r") as stream:
    file_paths = yaml.safe_load(stream)


if __name__ == "__main__":
    """Get the embeddings of description"""
    df, df_description = load_dataset(file_paths["dataset_path"])
    distilbert = SentenceTransformer(file_paths["model_name"])
    embeddings = Embeddings(distilbert, df_description).create_embeddings()

    """Recommend wines to User from their query"""
    user_query = input("Enter the details of your ideal choice wine:")
    ranking = Ranking(embeddings, distilbert, df)
    df_recommendation = ranking.recommend_wines(user_query)
