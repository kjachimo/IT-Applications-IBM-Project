import math

from src import Client
from src.mapper import Mapper

TARGET_SPEED = 50  # Target speed in km/h. Increasing this makes the car go faster but may reduce stability.
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
def drive_modular(c):
    S, R = c.S.d, c.R.d
    R["steer"] = calculate_steering(S)
    R["accel"] = calculate_throttle(S, R)
    R["brake"] = apply_brakes(S)
    R["accel"] = traction_control(S, R["accel"])
    R["gear"] = shift_gears(S)
    return


# ================= MAIN LOOP =================
if __name__ == "__main__":
    C = Client(p=3001)
    mapper = Mapper(print_image=True, live_plot=True)
    for step in range(C.maxSteps, 0, -1):
        C.get_servers_input()
        mapper.update(C.S.d)
        drive_modular(C)
        C.respond_to_server()
    mapper.save_json()
    mapper.close()
    C.shutdown()
