import math

from src import Client
from src.it_applications_ibm_project.prediction import SteeringModel

TARGET_SPEED = 70  # Target speed in km/h. Increasing this makes the car go faster but may reduce stability.
BRAKE_THRESHOLD = 0.9  # Angle threshold for braking. Lower values brake earlier.
GEAR_SPEEDS = [0, 20, 40, 80, 100, 180]  # Speed thresholds for gear shifting.
ENABLE_TRACTION_CONTROL = True  # Toggle traction control system.


STEERING_MODEL = SteeringModel()


def calculate_steering(S):
    baseline = (S["angle"] / math.pi) - (S["trackPos"] * 0.4)
    learned_adjustment = 0.15 * STEERING_MODEL.predict(S)
    steer = baseline + learned_adjustment
    return max(-1, min(1, steer))


def predict_track_difficulty(S):
    """Use model curvature detection to predict speed penalty.
    
    Higher curvature = sharper turn = more speed penalty.
    Returns:
        penalty (float): Speed reduction (0.0 to 50.0 km/h)
    """
    track = S.get("track") or []
    curvature = STEERING_MODEL._curvature_hint(track)
    
    # Convert curvature to speed penalty
    # curvature ranges from about -0.3 to +0.3, we map that to penalties
    curve_magnitude = abs(curvature)
    
    if curve_magnitude > 0.12:  # Sharp turn
        return 35.0
    elif curve_magnitude > 0.06:  # Moderate turn
        return 20.0
    elif curve_magnitude > 0.02:  # Gentle turn
        return 8.0
    else:
        return 0.0  # Straight


def calculate_throttle(S, R):
    steer_mag = abs(R["steer"])
    curve_penalty = steer_mag * 22.0
    track_difficulty_penalty = predict_track_difficulty(S)
    target_speed = TARGET_SPEED - curve_penalty - track_difficulty_penalty
    if S["speedX"] < target_speed:
        accel = min(1.0, R["accel"] + 0.25)
    else:
        accel = max(0.0, R["accel"] - 0.30)
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
