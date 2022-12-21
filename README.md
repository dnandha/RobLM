# RobLM

Supplemental code for the thesis: [Leveraging Large Language Models for Autonomous Task Planning](https://drive.google.com/file/d/1fsLtoKo4nMTdt_FUqjBVHWYn9-ezxoIc/view?usp=sharing).

## Installation
`conda env create -f env.yml` 
`conda activate roblm`

## Preprocessing
`python preprocess.py json_2.1.0/valid_unseen valid_unseen.json --cond knowledge_graph.dot`

## Training
`python roblm.py --train traindata.json [--chkpt_path <path_to_saved_model>] --model_path <path_to_new_model> --log <logdir>

## Evaluation
0. Complete evaluation
`python roblm.py --eval validdata.json --chkpt <path_to_saved_model>`
1. Task based evaluation: validation data split across files
`for infile in valid/*.json; do python roblm.py --eval $infile --chkpt <path_to_saved_model>; done`
