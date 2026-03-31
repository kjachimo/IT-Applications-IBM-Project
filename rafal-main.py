"""
RAFAL'S TRACK LEARNING AI DRIVER FOR TORCS
===========================================
Two-phase approach:
Phase 1 (LEARNING): Drive and record track geometry, speed, position
Phase 2 (OPTIMIZED): Calculate optimal racing line offline, then execute it

Simple, rock-solid baseline first. No complex logic - just work!
"""

import math
import json
import os
from src import Client

# ================= OPERATING MODES =================
LEARNING_MODE = True  # Set False to use learned track data
LEARNING_LAPS = 2  # Number of laps to record for learning
OUTPUT_FILE = "track_data.json"  # File to save learned track data

# ================= SIMPLE BASELINE PARAMETERS =================
# These are proven to work - based on snakeoil examples
TARGET_SPEED = 120  # km/h - conservative and reliable
LEARNING_TARGET_SPEED = 75  # km/h - slower and safer during learning laps
LEARNING_MAX_GEAR = 4  # cap gear in learning to keep pace controlled
STEER_GAIN = 25.0  # Steering response (tuned for stability)
TRACK_CENTER_GAIN = 0.10  # How hard to pull to center
LAUNCH_STEPS = 120  # Force launch for first frames if speed stays low
LAUNCH_MIN_SPEED = 15.0  # km/h threshold considered as "moving"

# ================= HELPER FUNCTIONS =================


def simple_steering(S):
    """
    Ultra-simple, proven steering logic:
    - Steer toward the direction car is pointing (angle)
    - Steer back toward track center (trackPos)
    """
    # Steer proportional to angle (turn into the slide)
    steer = S["angle"] * STEER_GAIN / math.pi

    # Steer back to center
    steer -= S["trackPos"] * TRACK_CENTER_GAIN

    # Keep in range
    return max(-1.0, min(1.0, steer))


def simple_throttle(S, R, target_speed):
    """
    Ultra-simple throttle: just maintain target speed
    """
    accel = R["accel"]

    # If going slower than target, accelerate
    if S["speedX"] < target_speed:
        accel = min(1.0, accel + 0.05)
    else:
        # If going faster, back off
        accel = max(0.0, accel - 0.02)

    # At very low speeds, boost hard to get moving
    if S["speedX"] < 10:
        accel = min(1.0, accel + 0.5)

    return max(0.0, min(1.0, accel))


def simple_brake(S):
    """
    Only brake if at extreme angle (about to crash)
    """
    # Never brake at low speed; this can prevent launch from standstill.
    if S.get("speedX", 0.0) < 25.0:
        return 0.0
    if abs(S["angle"]) > 0.95:  # High angle = braking time
        return 0.4
    return 0.0


def simple_gear(S):
    """
    Basic gear shifting based on speed
    """
    speed = S["speedX"]

    if speed < 20:
        return 1
    elif speed < 40:
        return 2
    elif speed < 80:
        return 3
    elif speed < 120:
        return 4
    elif speed < 180:
        return 5
    else:
        return 6


# ================= TRACK LEARNING =================


class TrackRecorder:
    """Records track data during learning phase."""

    def __init__(self):
        self.data = {
            "waypoints": [],
            "lap_count": 0,
            "lap_distances": [],
        }
        self.last_distance = 0
        self.last_dist_from_start = None
        self.lap_started = False
        self.current_lap_waypoints = []

    def record(self, S):
        """Record current state as a track waypoint."""
        waypoint = {
            "position": float(S["trackPos"]),
            "angle": float(S["angle"]),
            "speed_optimal": float(TARGET_SPEED),  # We'll refine this later
            "track_sensors": [float(x) for x in S["track"]],
            "distance_from_start": float(S["distFromStart"]),
        }
        self.current_lap_waypoints.append(waypoint)

    def check_lap_complete(self, S):
        """Check if we've completed a lap."""
        current_distance = S["distRaced"]
        current_from_start = float(S.get("distFromStart", 0.0))

        # Lap wrap detection must use distFromStart, not distRaced.
        # distFromStart wraps from near track_length back to ~0 at start line.
        if (
            self.last_dist_from_start is not None
            and current_distance > 200
            and current_from_start < self.last_dist_from_start - 100
        ):
            self.data["lap_count"] += 1
            self.data["lap_distances"].append(self.last_distance)
            self.data["waypoints"].extend(self.current_lap_waypoints)

            print(
                f"✓ LAP {self.data['lap_count']} COMPLETE - Distance: {self.last_distance:.1f}m"
            )

            self.current_lap_waypoints = []

        self.lap_started = True
        self.last_distance = current_distance
        self.last_dist_from_start = current_from_start

    def save(self, filename=OUTPUT_FILE):
        """Save recorded data to file."""
        # Save completed laps and current partial lap so no learning data is lost.
        payload = dict(self.data)
        payload["waypoints"] = self.data["waypoints"] + self.current_lap_waypoints
        payload["partial_lap_waypoints"] = len(self.current_lap_waypoints)
        with open(filename, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"✓ Track data saved to {filename}")
        print(f"  - Waypoints: {len(payload['waypoints'])}")
        print(f"  - Laps: {payload['lap_count']}")


class TrackAnalyzer:
    """Analyzes recorded track data to find optimal speeds."""

    @staticmethod
    def load(filename=OUTPUT_FILE):
        """Load recorded track data."""
        if not os.path.exists(filename):
            return None

        with open(filename, "r") as f:
            return json.load(f)

    @staticmethod
    def calculate_optimal_speeds(track_data):
        """Calculate optimal speed for each track position."""
        # Group waypoints by position on track (using distFromStart)
        waypoints_by_distance = {}

        for wp in track_data["waypoints"]:
            dist = wp["distance_from_start"]
            dist_key = round(dist / 10) * 10  # Group by 10m sections

            if dist_key not in waypoints_by_distance:
                waypoints_by_distance[dist_key] = []

            waypoints_by_distance[dist_key].append(wp)

        # Calculate average angle and optimal speed for each section
        optimal_speeds = {}
        for dist_key, waypoints in waypoints_by_distance.items():
            avg_angle = sum(abs(wp["angle"]) for wp in waypoints) / len(waypoints)

            # Higher angle = tighter turn = slower speed needed
            # Speed(angle) = BASE - (angle / π) * FACTOR
            factor = 100  # Speed reduction factor
            optimal_speed = TARGET_SPEED - (avg_angle / math.pi) * factor
            optimal_speed = max(50, min(TARGET_SPEED, optimal_speed))  # Clamp

            optimal_speeds[dist_key] = {
                "distance": dist_key,
                "avg_angle": avg_angle,
                "optimal_speed": optimal_speed,
                "wp_count": len(waypoints),
            }

        return optimal_speeds


# ================= MAIN DRIVE FUNCTION =================


class RafalDriver:
    """Simple, reliable driver with learning capability."""

    def __init__(self, learning_mode=True):
        self.learning_mode = learning_mode
        self.recorder = TrackRecorder() if learning_mode else None
        self.lap_count = 0
        self.prev_distance = 0
        self.launch_steps_left = LAUNCH_STEPS

    def drive(self, client):
        """Main driving logic."""
        S = client.S.d
        R = client.R.d

        # ===== GUARANTEED LAUNCH PHASE =====
        speed = S.get("speedX", 0.0)
        if self.launch_steps_left > 0 and speed < LAUNCH_MIN_SPEED:
            self.launch_steps_left -= 1
            R["steer"] = max(-0.5, min(0.5, simple_steering(S)))
            R["accel"] = 1.0
            R["brake"] = 0.0
            R["gear"] = 1
            R["clutch"] = 0.0
            return

        # ===== LEARNING PHASE =====
        if self.learning_mode and self.recorder:
            self.recorder.check_lap_complete(S)

            # Stop learning after N laps
            if self.recorder.data["lap_count"] >= LEARNING_LAPS:
                print("\n" + "=" * 60)
                print("LEARNING COMPLETE!")
                print("=" * 60)
                self.recorder.save()

                # Analyze and print results
                analyzer = TrackAnalyzer()
                optimal_data = analyzer.calculate_optimal_speeds(self.recorder.data)
                print("\nOptimal speeds by track section:")
                for dist_key in sorted(optimal_data.keys())[:10]:
                    info = optimal_data[dist_key]
                    print(
                        f"  {info['distance']:4.0f}m: {info['optimal_speed']:5.1f} km/h (angle: {info['avg_angle']:.3f})"
                    )

                # Stop the simulation
                self.learning_mode = False
                # Note: Would normally exit here, but continue driving with learned data

            self.recorder.record(S)

        # ===== SIMPLE DRIVING CONTROL =====
        target_speed = LEARNING_TARGET_SPEED if self.learning_mode else TARGET_SPEED
        R["steer"] = simple_steering(S)
        R["accel"] = simple_throttle(S, R, target_speed)
        R["brake"] = simple_brake(S)
        R["gear"] = simple_gear(S)
        if self.learning_mode:
            R["gear"] = min(R["gear"], LEARNING_MAX_GEAR)
        R["clutch"] = 0.0

    def finalize(self):
        """Persist any recorded data when program exits early or on completion."""
        if self.recorder is None:
            return
        if self.recorder.data["waypoints"] or self.recorder.current_lap_waypoints:
            self.recorder.save()


# ================= MAIN LOOP =================

if __name__ == "__main__":
    print("=" * 60)
    print("RAFAL'S TRACK LEARNING AI DRIVER")
    print("=" * 60)

    if LEARNING_MODE:
        print(f"MODE: LEARNING (will record {LEARNING_LAPS} laps)")
        print(f"Output: {OUTPUT_FILE}")
    else:
        print("MODE: OPTIMIZED (using learned track data)")

    print("=" * 60)
    print()

    C = None
    driver = None
    try:
        # Initialize client
        C = Client(p=3001)
        driver = RafalDriver(learning_mode=LEARNING_MODE)

        print("Connected to TORCS on port 3001")
        print(f"Starting simulation ({C.maxSteps} steps)...")
        print()

        # Main loop
        step_count = 0
        for step in range(C.maxSteps, 0, -1):
            C.get_servers_input()
            driver.drive(C)
            C.respond_to_server()

            step_count += 1

            # Status every 250 steps
            if step_count % 250 == 0:
                speed = C.S.d.get("speedX", 0)
                pos = C.S.d.get("distRaced", 0)
                fuel = C.S.d.get("fuel", 0)
                print(
                    f"Step {step_count:6d} | Speed: {speed:6.1f} km/h | Distance: {pos:8.1f}m | Fuel: {fuel:6.1f}L"
                )

        print()
        print("=" * 60)
        print("SIMULATION COMPLETE!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n[INFO] Simulation interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()
    finally:
        if driver is not None:
            driver.finalize()
        if C is not None:
            C.shutdown()
        print("[INFO] Connection closed.")
