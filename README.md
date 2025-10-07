# 🥚🍳 Event-Grounding Graph (EGG)

> **:warning: Warning**<br>
> This repository and some of its requirements are under active development, so bugs and breaking changes are expected.

## 🚀 Overview

**Abstract:** A fundamental aspect for building intelligent autonomous
robots that can assist humans in their daily lives is the
construction of rich environmental representations. While advances in
semantic scene representations have enriched robotic
scene understanding, current approaches lack a connection
between the spatial features and dynamic event interactions.
In this work, we introduce event-grounding graph (EGG), a
framework grounding event interactions to spatial features of
a scene. This representation allows robots to perceive, reason,
and respond to complex spatio-temporal queries. Experiments
using real robotic data demonstrate EGG’s capability to re-
trieve relevant information and respond accurately to human
inquiries concerning the environment

**Authors:** Anonymous

The prompts that are used for Generative AI are available [here](./Appendix.md)

## 🤖 Requirements

We tested EGG on a laptop with an RTX 3070 GPU Mobile.

🐳 We highly recommend using Docker for deploying EGG. We provide pre-built Dockerfiles here (https://github.com/aalto-intelligent-robotics/EGG-docker)

If you do not want to use Docker for some reason, EGG was tested on Ubuntu 20.04 with ROS Noetic installed. Follow the instructions [here](https://wiki.ros.org/ROS/Installation) if you do not yet have it on your system. The other requirements are:

- [VideoRefer](https://github.com/DAMO-NLP-SG/VideoRefer) (if you want to automatically generate video captions, otherwise you can use the provided ground truth data)
- OpenAI AI API key (for graph pruning, evaluation, and generating image captions)

## 🧰 Building EGG

### 🐳 Setting up with Docker

Clone the Docker repo:

```bash
git clone https://github.com/aalto-intelligent-robotics/EGG-docker.git
cd EGG-docker/
```

Create these directories:

- `logs`: For EGG's output (debugging)
- `data`: For data
- `bags`: Put your ROS bags here (or create a symlink)

If you want to generate video captions automatically, you need to set up VideoRefer. The dependencies are automatically installed with Docker, so you only need to clone the source code.

```bash
mkdir third_party/
cd third_party
git clone git@github.com:DAMO-NLP-SG/VideoRefer.git
cd ..
```

Now build the docker image:
```bash
docker compose build base
```

Go grab yourself a coffee because this can take a while ☕

To start a container:

```bash
docker compose up base -d
docker exec -it egg_base bash
```

# Data processing
This part of the repository is anonymized during the double-blind review process.

## 🔥 Quickstart
To start with EGG, you need to set up the data as folowed:

*Note: The dataset will be made public after the double-blind review process*

To build EGG, use one of the following:
```bash
cd egg/app
# To build a graph from ground truth
python3 build_graph.py
# To build a graph with guided captioning (Requires GPU!)
python3 build_graph.py -a
# To build a graph with unguided captioning (Requires GPU!)
python3 build_graph.py -a -u
```

The result will be a json file, e.g., "graph_gt.json"

To visualize EGG, make sure you have open3d installed and run:
```bash
cd egg/app
python3 egg_visualizer.py
```

To replicate the information retrieval experiments:
```bash
cd egg/app
# Change the strategy, the possible values are ['pruning_unified', 'pruning_unified_no_edge', 'spatial', 'event', 'no_edge', 'full_unified']
python3 eval.py -s pruning_unified -t 1
```
