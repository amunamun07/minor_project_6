class Embeddings:
    def __init__(self, distilbert_model, df_description):
        self.distilbert_model = distilbert_model
        self.df_description = df_description

    def create_embeddings(self):
        """Creates Embeddings from the given 'description' dataframe

        Returns:
            embeddings(tensor): A tensors of 'df_description' embeddings
        """
        embeddings = (
            self.distilbert_model.encode(
                self.df_description["description"], convert_to_tensor=True
            )
            .cpu()
            .numpy()
        )
        return embeddings
