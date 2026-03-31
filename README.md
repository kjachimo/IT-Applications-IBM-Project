# IT Applications IBM Project

## Installation

1. Clone the repository:

        git clone git@github.com:kjachimo/IT-Applications-IBM-Project.git

2. Extract the IBM zip into `IT-Applications-IBM-Project/`
3. Remove `IT-Applications-IBM-Project/gym_torcs/.git`

## Train AI Driver (PyTorch DDPG)

This repository now includes an automatic training entrypoint based on the
TORCS DDPG tutorial workflow, implemented on top of the `src` client stack.

### Prerequisites (Linux)

1. Install TORCS and ensure `torcs` is in PATH.
2. Install `xautomation` (`xte`) for menu automation.
3. Use a desktop/X11 session where TORCS windows can receive key presses.
4. Install Python dependencies:

        pip install -r requirements.txt

### Run training

From repository root:

        python rafal-train.py --track-name corkscrew --track-category road

If you run TORCS with Wine (for example `wtorcs.exe` inside the `torcs/`
folder), use:

        python rafal-train.py --torcs-command "wine wtorcs.exe" --torcs-dir torcs --track-name corkscrew --track-category road

Useful flags:

        --episodes 200
        --max-steps 2000
        --relaunch-every 3
        --checkpoint-dir checkpoints
        --checkpoint-every 10
        --torcs-command "wine wtorcs.exe"
        --torcs-dir torcs
        --autostart-script gym_torcs/autostart.sh
        --stop-torcs-on-exit

### Notes

1. The startup script is automated via `autostart.sh`/`gym_torcs/autostart.sh`.
2. The first target is the corkscrew track with default TORCS car settings.
3. If your local TORCS menu flow differs, adjust the autostart script key
   sequence.
