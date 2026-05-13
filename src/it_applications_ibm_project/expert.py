import math

from it_applications_ibm_project.server_state import SensorData
from it_applications_ibm_project.driver_action import ActionData

TARGET_SPEED = 100  # Target speed in km/h. Increasing this makes the car go faster but may reduce stability.
STEER_GAIN = (
    30  # Steering sensitivity. Higher values make the car turn more aggressively.
)
CENTERING_GAIN = (
    0.20  # How strongly the car corrects its position toward the center of the track.
)
BRAKE_THRESHOLD = 0.9  # Angle threshold for braking. Lower values brake earlier.
GEAR_SPEEDS = [0, 20, 40, 80, 100, 180]  # Speed thresholds for gear shifting.
ENABLE_TRACTION_CONTROL = True  # Toggle traction control system.


def calculate_steering(S):
    steer = (S["angle"] * STEER_GAIN / math.pi) - (S["trackPos"] * CENTERING_GAIN)
    return max(-1, min(1, steer))


def calculate_throttle(S, R):
    if S["speedX"] < TARGET_SPEED - (R["steer"] * 2.5):
        accel = min(1.0, R["accel"] + 0.4)
    else:
        accel = max(0.0, R["accel"] - 0.2)
    if S["speedX"] < 10:
        accel += 1 / (S["speedX"] + 0.1)
    return max(0.0, min(1.0, accel))


def apply_brakes(S):
    return 0.3 if abs(S["angle"]) > BRAKE_THRESHOLD else 0.0


def shift_gears(S):
    gear = 1
    for i, speed in enumerate(GEAR_SPEEDS):
        if S["speedX"] > speed:
            gear = i + 1
    return min(gear, 6)


def traction_control(S, accel):
    if ENABLE_TRACTION_CONTROL:
        if (
            (S["wheelSpinVel"][2] + S["wheelSpinVel"][3])
            - (S["wheelSpinVel"][0] + S["wheelSpinVel"][1])
        ) > 2:
            accel -= 0.1
    return max(0.0, accel)


# ================= MAIN DRIVE FUNCTION =================
def drive_modular(state: SensorData, action: ActionData) -> ActionData:
    R = action
    R["steer"] = calculate_steering(state)
    R["accel"] = calculate_throttle(state, R)
    R["brake"] = apply_brakes(state)
    R["accel"] = traction_control(state, R["accel"])
    return R
