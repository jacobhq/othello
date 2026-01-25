# othello

Othello is a platform similar to Chess.com, but for Othello. It was built for my A-Level Computer Science NEA project, the goal being to use everything I have ever learnt in Computer Science. So far, the Othello website is available, but as you are reading this, I am in the process of training an AI model to play against. After this has been trained and integrated into the website, development will pause and the project will be submitted to my exam board, OCR. After I have completed my A-Levels, I may return to this project, and implement multiplayer.

The project is currently running on the smallest and cheapest Fly.io VMs, and the models are training on free credits from various cloud providers on NVIDIA L4s and NVIDIA A100 40GBs. The model is very small, and so will be able to run in the browser, and I plan to move the web server onto a cluster of Raspberry Pis.

A detailed write-up has been written as part of my NEA, and is available [here](https://example.com). If you would like to contact me regarding this project, then please send me an email at the address on my profile page.

---

I will give a brief explanation of what the different modules in the project do however:

- `crates/`
  - `othello` - Core Othello implementation with BitBoards
  - `othello-api` - Axum API with hand-rolled artisanal auth, interfaces with the frontend and a postgres database
  - `othello-demo` - CLI application for testing out Othello in the terminal
  - `othello-learn` - CLI responsible for running the training loop (`othello-self-play` and `othello-training`), supports configuration and resuming of training
  - `othello-self-play` - Policy guided MCTS data generation into a custom binary format
  - `othello-wasm` - Wasm bindings to the core `othello` crate
- `packages/`
  - `othello-training` - PyTorch CNN with policy and value heads, ONNX export, data loading
- `apps/`
  - `othello-frontend` - React frontend for the whole Othello platform
