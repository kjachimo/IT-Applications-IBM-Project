from .utils import clip, bargraph


class DriverAction:
    """What the driver is intending to do (i.e. send to the server).
    Composes something like this for the server:
    (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus 0)(meta 0) or
    (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus -90 -45 0 45 90)(meta 0)"""

    def __init__(self):
        self.actionstr = str()
        self.d = {
            "accel": 0.2,
            "brake": 0,
            "clutch": 0,
            "gear": 1,
            "steer": 0,
            "focus": [-90, -45, 0, 45, 90],
            "meta": 0,
        }

    def clip_to_limits(self):
        """There pretty much is never a reason to send the server
        something like (steer 9483.323). This comes up all the time
        and it's probably just more sensible to always clip it than to
        worry about when to. The "clip" command is still a snakeoil
        utility function, but it should be used only for non standard
        things or non obvious limits (limit the steering to the left,
        for example). For normal limits, simply don't worry about it."""
        self.d["steer"] = clip(self.d["steer"], -1, 1)
        self.d["brake"] = clip(self.d["brake"], 0, 1)
        self.d["accel"] = clip(self.d["accel"], 0, 1)
        self.d["clutch"] = clip(self.d["clutch"], 0, 1)
        if self.d["gear"] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.d["gear"] = 0
        if self.d["meta"] not in [0, 1]:
            self.d["meta"] = 0
        if (
            type(self.d["focus"]) is not list
            or min(self.d["focus"]) < -180
            or max(self.d["focus"]) > 180
        ):
            self.d["focus"] = 0

    def __repr__(self):
        self.clip_to_limits()
        out = str()
        for k in self.d:
            out += "(" + k + " "
            v = self.d[k]
            if not type(v) is list:
                out += "%.3f" % v
            else:
                out += " ".join([str(x) for x in v])
            out += ")"
        return out
        return out + "\n"

    def fancyout(self):
        """Specialty output for useful monitoring of bot's effectors."""
        out = str()
        od = self.d.copy()
        od.pop("gear", "")  # Not interesting.
        od.pop("meta", "")  # Not interesting.
        od.pop("focus", "")  # Not interesting. Yet.
        for k in sorted(od):
            if k == "clutch" or k == "brake" or k == "accel":
                strout = ""
                strout = "%6.3f %s" % (od[k], bargraph(od[k], 0, 1, 50, k[0].upper()))
            elif k == "steer":  # Reverse the graph to make sense.
                strout = "%6.3f %s" % (od[k], bargraph(od[k] * -1, -1, 1, 50, "S"))
            else:
                strout = str(od[k])
            out += "%s: %s\n" % (k, strout)
        return out
