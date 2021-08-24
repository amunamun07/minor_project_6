# Minor_Project_6

The objective for this project is to create a ranking of wines based on similarity scores and recommend user their 15 ideal choices of wines.

### Folder Description:
#### 1. Notebook
- Contains ranking.ipnyb which will show step by step cell process for fulfilling the project objective i.e. ranking and recommending wines based on their query.

### Pre-requisites:
- pipenv(sudo apt-get pipenv)
- Hardware accelerator = GPU (Highly Recommended)

### How to run this repository:
#### Step 1: Setup 
- Clone the repo.
- In the folder, run 'pipenv shell' 
- Then run 'pipenv install'

#### Step 2: Setup folders
- create a dataset folder
- Add dataset from https://www.kaggle.com/zynicide/wine-reviews
- You will find a csv dataset with 130k datas. For simplicilty you can only keep the 20k data rows.
- Set the 'dataset_path' in config.yaml

#### Step 3.1: Run via. py file
- In the project folder, execute 'python main.py'

#### Step 3.2: Run via. Notebook
- In the project folder, execute 'pipenv run jupyter notebook'.
- Open "Notebook" folder > ranking.ipnyb
- Run each cell as instructed in the notebook.

#### Output:
- You will be asked to describe your wine taste. You can enter something like "Sweet, red wine fine texture", or you can go ahead and add your own details that describes a wine. 
- The final output is the list of ranked wines based on similarity and also 15 recommended wines in a dataframe.
