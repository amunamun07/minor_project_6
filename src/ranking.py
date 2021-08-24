import nmslib
import pandas as pd


class Ranking:
    def __init__(self, embeddings, distilbert_model, df):
        """Initializing ranking class with the following variables and objects

        Args:
            distilbert_model(object): User given query
            embeddings(tensor): A tensors of 'df_description' embeddings
            df(dataframe): A dataframe of the original dataset
        """
        self.distilbert_model = distilbert_model
        self.embeddings = embeddings
        self.df = df

    @staticmethod
    def create_nmslib_search_index(embeddings):
        """Initialize a new index, using a HNSW index on Cosine Similarity

        Args:
            embeddings(tensor): A tensors of 'df_description' embeddings
        Returns:
            embeddings_index(object): Search indexes For the embeddings
        """
        embeddings_index = nmslib.init(
            method="hnsw", space="cosinesimil", data_type="dense_vector"
        )
        embeddings_index.addDataPointBatch(embeddings)

        post_processing = {"post": 2}
        embeddings_index.createIndex(post_processing, print_progress=True)
        return embeddings_index

    def recommend_wines(self, user_query):
        """Recommends wine based on user query and similarity score

        Args:
            user_query(str): User given query
        Returns:
            df_recommended_wines(dataframe): A dataframe of the recommendation
        """
        embeddings_index = self.create_nmslib_search_index(self.embeddings)
        query = (
            self.distilbert_model.encode([user_query], convert_to_tensor=True)
            .cpu()
            .numpy()
        )
        nearest_neighbours = 500
        ids, distances = embeddings_index.knnQuery(query, k=nearest_neighbours)
        ranks = []
        for id, distance in zip(ids, distances):
            ranks.append(
                {
                    "country": self.df.country.values[id],
                    "winery": self.df.winery.values[id],
                    "variety": self.df.variety.values[id],
                    "price": self.df.price.values[id],
                    "similarity_score": distance,
                }
            )
        df_ranked_wines = pd.DataFrame(ranks)
        no_of_recommendation = 15
        df_recommended_wines = df_ranked_wines.nlargest(
            no_of_recommendation, "similarity_score"
        )
        return df_recommended_wines
