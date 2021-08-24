import pandas as pd


def load_dataset(dataset_path):
    """Loads the wine dataset from the given path

    Args:
        dataset_path(str): Path of the dataset
    Returns:
        df(dataframe): A dataframe of the dataset
        df_description(dataframe): A dataframe of only the 'description' column
    """
    df = pd.read_csv(
        dataset_path, sep=",", engine="python", quotechar='"', error_bad_lines=False
    )

    df_description = df.copy()
    df_description.drop(
        df_description.columns.difference(["description"]), axis=1, inplace=True
    )

    return df, df_description
