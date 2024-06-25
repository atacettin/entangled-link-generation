# Adaptable Resource Generation Protocols For Quantum Networks
## About the project
This project is part of the Bachelor's Thesis "Adaptable Resource Generation Protocols For Quantum Networks." This project contains a simulation of a quantum network carrying out an entanglement generation process
for multiple links.

`LinkGenerationEnv.py` contains the continuous and discrete action space simulation

`Heuristics.py` contains our heuristic, the single action policy, and the random policy 

`HeuristicTesting.ipynb` contains some utility to test the heuristic

`Visualistions.ipynb` contains methods to visualise results.


To train an RL model, see `run_experiments.sh` for an example. Available parameters can be found in `train_c51.py` and `train_REINFORCE.py`.

Saved models can be found in `saved_models`
