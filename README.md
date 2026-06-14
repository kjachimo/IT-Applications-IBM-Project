# IT Applications IBM Project

AI driver for Torcs using SAC (Soft Actor-Critic).

## Best times

* **Best Time (mm:ss:ms):** 1:49:21
* **Video:** https://drive.google.com/file/d/1_J27Yw8RZXu1hCwk3mH0uK6YP0HaW5Ww/view?usp=sharing

## Installation

1. Clone the repository:

        git clone git@github.com:kjachimo/IT-Applications-IBM-Project.git

2. Extract the IBM zip into `IT-Applications-IBM-Project/`
3. Remove `IT-Applications-IBM-Project/gym_torcs/.git`

## Usage

To try the currently best model on Corkscrew:

```bash
uv run eval_sac.py --model-path checkpoints/checkpoints_sac/exp1/sac_best_distance.zip
```
