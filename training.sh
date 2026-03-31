#!/bin/bash

uv run python rafal-train.py \
  --track-name corkscrew \
  --track-category road \
  --torcs-command "wine wtorcs.exe" \
  --torcs-dir torcs \
  --episodes 200 \
  --max-steps 2000 \
  --relaunch-every 6 \
  --checkpoint-every 6 \
  --resume ./checkpoints/ddpg_ep0010.pt \
  --stop-torcs-on-exit
