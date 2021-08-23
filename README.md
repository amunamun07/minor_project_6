# Minor_Project_6

The objective for this project is to create a ranking of wines based on similarity scores and recommend user their 15 ideal choices of wines.

### Folder Description:
#### 1. Notebook
- Contains ranking.ipnyb which will show step by step cell process for fulfilling the project objective i.e. ranking and recommending wines based on their query.

### How to run this repository:
Pre-requisits: Install pipenv(sudo apt-get pipenv) and clone the repository(git clone).
#### Step 1: Setup pipenv
- Go to the project folder.
- Open terminal
- pipenv shell

#### Step 2: Setup folders
- create a dataset folder
- Add dataset from https://www.kaggle.com/zynicide/wine-reviews
- You will find a csv dataset with 130k datas. For simplicilty you can only keep the 20k data rows.
- Also, rename the csv to wine_dataset.csv

#### Step 3: Run
- Open the notebook by typing 'pipenv run jupyter notebook' on terminal.
- Jupyter notebook will open up, Open "Notebook" folder > ranking.ipnyb
- Run each cell as instructed in the notebook.

#### Output:
- The final output is the list of ranked wines based on similarity and also 15 recommended wines in a dataframe.
