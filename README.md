# ASLRecognition

Collection of Python scripts for ASL fingerspelling recognition pipeline

## Installation

Clone this repo and install requirements outlined in requirements.txt. Virtual environment is optional.

```bash
git clone https://github.com/joren015/ASLRecognition.git
python -m venv .venv
source ./.venv/bin/activate
python -m pip install -r requirements.txt
```

## Usage

Upload either of the two notebook files (.ipynb) in a Google Colab notebook. Run the notebook from start to finish. This will clone the repo, extract the needed files, and run the pipeline example. HomemadeDataset.ipynb is for the still images dataset and HomemadeDataset2.ipynb is for the video dataset.

If you would like to run the entier pipeline locally, HomemadeDataset.py and HomemadeDataset2.py executes the preprocessing, labeling, and training for both the still images dataset and the video dataset.

