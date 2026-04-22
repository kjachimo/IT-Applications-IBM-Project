import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from itertools import product


def time_to_sec(t):
    if isinstance(t, str) and ":" in t:
        t = t.replace(",", ".")
        m, s = t.split(":")
        return int(m) * 60 + float(s)
    return np.nan


data = [
    [70, 30, 0.2, 0.3, [0, 20, 40, 80, 100, 180], 1, "03:07.10", "03:03.61"],
    [70, 30, 0.2, 0.3, [0, 20, 40, 80, 100, 180], 0, "03:05.10", "03:02.52"],
    [70, 18, 0.35, 0.55, [0, 25, 45, 70, 95, 150], 1, "03:06.44", "03:04.83"],
    [70, 18, 0.35, 0.55, [0, 25, 45, 70, 95, 150], 0, "03:06.23", "03:04.81"],
    [75, 20, 0.32, 0.45, [0, 25, 50, 80, 115, 180], 1, "02:55.28", "02:53.63"],
    [75, 20, 0.32, 0.45, [0, 25, 50, 80, 115, 180], 0, "02:55.08", "02:53.64"],
    [75, 20, 0.35, 0.40, [0, 25, 50, 80, 115, 180], 0, "02:55.13", "02:53.60"],
]

df = pd.DataFrame(
    data,
    columns=[
        "TARGET_SPEED",
        "STEER_GAIN",
        "CENTERING_GAIN",
        "BRAKE_THRESHOLD",
        "GEAR_SPEEDS",
        "TC",
        "lap1",
        "lap2",
    ],
)

df["lap2_sec"] = df["lap2"].apply(time_to_sec)
df = df.dropna(subset=["lap2_sec"])

gear_df = pd.DataFrame(df["GEAR_SPEEDS"].tolist(), columns=[f"g{i}" for i in range(6)])
df = pd.concat([df.drop(columns=["GEAR_SPEEDS", "lap1", "lap2"]), gear_df], axis=1)

X = df.drop(columns=["lap2_sec"])
y = df["lap2_sec"]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

results = []

for ts, sg, cg, bt, tc in product(
    [70, 75, 77, 80], [18, 20, 22], [0.30, 0.32, 0.35], [0.35, 0.40, 0.45], [0, 1]
):
    gears = [0, 25, 50, 80, 115, 180]
    row = [ts, sg, cg, bt, tc] + gears
    pred = model.predict([row])[0]

    results.append(
        {
            "pred_time": pred,
            "TARGET_SPEED": ts,
            "STEER_GAIN": sg,
            "CENTERING_GAIN": cg,
            "BRAKE_THRESHOLD": bt,
            "TC": tc,
        }
    )

results_sorted = sorted(results, key=lambda x: x["pred_time"])

top5 = results_sorted[:5]

for i, r in enumerate(top5, 1):
    print(f"\n#{i}")
    print(f"czas: {r['pred_time']:.2f} s")
    print(
        f"TS={r['TARGET_SPEED']}, SG={r['STEER_GAIN']}, CG={r['CENTERING_GAIN']}, BT={r['BRAKE_THRESHOLD']}, TC={r['TC']}"
    )
