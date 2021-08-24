import yaml
from sentence_transformers import SentenceTransformer

from src import Ranking, Embeddings, load_dataset

with open("config.yaml", "r") as stream:
    file_paths = yaml.safe_load(stream)


def load_model_and_embeddings():
    distilbert = SentenceTransformer(file_paths["model_name"])
    embeddings = Embeddings(distilbert, df_description).create_embeddings()
    return distilbert, embeddings


def main(user_query_, distilbert, embeddings):
    ranking = Ranking(embeddings, distilbert, df)
    df_recommendation = ranking.recommend_wines(user_query_)
    return df_recommendation


if __name__ == "__main__":
    """Get the embeddings of description"""
    df, df_description = load_dataset(file_paths["dataset_path"])
    distilbert_model, embedding = load_model_and_embeddings()

    """Recommend wines to User from their query"""
    user_query = input("Enter the details of your ideal choice wine:")
    df_recommend = main(user_query, distilbert_model, embedding)
    print(df_recommend)
