# SRR-DDI: A Drug-Drug Interaction Prediction Model with Substructure Refined Representation Learning based on Self-Attention Mechanism

Scenario Requirements
torch==1.9.0
python==3.7.16
dgl==0.6.1
numpy==1.20.0
pandas==1.3.5
rdkit==2020.9.5.2
torch-geometric==2.2.0

Running
First, run data_pre.py to generate three-folds of data
python data_pre.py -d drugbank -o all
Second, run train.py to train SRR-DDI
python train.py
If you want to store models and logs, using --save_model parameters 
