# Coming up

## Installation
`conda env create -f env.yml` 
`conda activate roblm`

## Preprocessing
`python preprocess.py json_2.1.0/valid_unseen valid_unseen.json --cond knowledge_graph.dot`

## Training
`python roblm.py --train traindata.json --chkpt_path checkpoints/model.pt`
