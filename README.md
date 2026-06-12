# AlphaZero like model for 3x3 Go (Pytorch)

This repository contains a pytorch implementation of a residual net reinforcement learning model to learn the strategy for 3x3 Go through self play.

## Structure

* `environment.py`: Contains the `RealGoGame` class. Manages the board state, legal moves, Ko rule checks, and territory scoring (Area Scoring).
* `model.py`: Contains the Pytorch AlphaZeroNet model and state-to-tensor encoding
* `mcts.py`: Monte-Carlo Tree Search. How the engine tests various moves from a given board state.
* `trainer.py`: Has the main loop, complete_training_loop(). Orchestrates self-play and learning.
* 'GoNotebook.ipynb': Jupyter notebook that calls the training and confirms the results.

## Results & Convergence

3x3 Go was chosen so the convergence can be verified on sight (playing in the center, `(1, 1)`), with training succeeding in around 500 games (~5 min on my laptop's cpu).
The notebook also shows how the model plays against an oponnent playing random moves.
```text
Network Evaluation of Empty Board: 0.8917
>> VERDICT: It correctly knows Black has the advantage.

Policy Probability Map:
[[0.    0.    0.   ]
 [0.    0.997 0.002]
 [0.    0.    0.   ]]
```

## How to run
This project was built using a Docker container. After cloning the repository...

Replicate by building the docker image corresponding to the dockerfile in the repo (which in turn references requirements.txt).
The docker build takes a while due to pytorch (up to 20 min depending on network speed).

Once the docker image is built, run the jupyter notebook using your docker image.This can be a little tricky, but on linux my bash command looked something like:
docker run -it \
  -p 8888:8888 \
  -v "/home/.../code_location:/app" \
  -v "/home/user_name/.jupyter_home:/home/jupyter" \
  -e HOME=/home/jupyter \
  -w /app \
  --user $(id -u):$(id -g) \
  --env-file "/home/.../environment_file.location/.env" \
  your-built-docker-image-name

Something similar should work on windows using the terminal or powershell.
