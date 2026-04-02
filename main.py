import math

from src import Client

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


class DynamicGainAdjuster:
    def __init__(self):
        self._kp = 1.5
        self._ki = 0.01
        self._kd = 0.8
        self._prev_error = 0.0

    def adjust(self, curvatures: list[float], speed: float) -> tuple[float, float]:
        if not curvatures:
            return self._kp, self._ki

        max_lookahead = min(len(curvatures), 4)
        lookahead_curv = sum(curvatures[-max_lookahead:]) / max_lookahead

        anticipatory_gain = max(0.0, lookahead_curv * 1.2)
        new_kp = min(self._kp + anticipatory_gain, 3.5)

        ki_adj = 0.0 if speed <= 10.0 else -0.05
        new_ki = max(0.0, self._ki + ki_adj)
        return new_kp, new_ki


def estimate_curvature_profile(track_sensors: list[float]) -> list[float]:
    if len(track_sensors) < 19:
        return [0.0]

    center = 9
    pairs = [(1, 1), (2, 2), (3, 3), (4, 4)]
    curvatures = []
    for l_off, r_off in pairs:
        left = track_sensors[center - l_off]
        right = track_sensors[center + r_off]
        denom = max(left + right, 1e-6)
        curvatures.append((right - left) / denom)
    return curvatures


GAIN_ADJUSTER = DynamicGainAdjuster()


def calculate_steering(S):
    speed_mps = S["speedX"] / 3.6
    curvatures = estimate_curvature_profile(S["track"])
    kp, ki = GAIN_ADJUSTER.adjust(curvatures, speed_mps)

    dynamic_centering = CENTERING_GAIN + ki
    steer = (S["angle"] * kp / math.pi) - (S["trackPos"] * dynamic_centering)
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
    for step in range(C.maxSteps, 0, -1):
        C.get_servers_input()
        drive_modular(C)
        C.respond_to_server()
    C.shutdown()
