"""Offline SO-101 LeRobot IK check. Reads JSON lines, never connects to hardware."""
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
MODEL = ROOT / "vendor" / "so101_kinematics_only.urdf"
spec = importlib.util.spec_from_file_location("lerobot_kinematics", ROOT / "vendor" / "lerobot043" / "lerobot" / "model" / "kinematics.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
root = ET.parse(MODEL).getroot()
bounds = {}
for joint in root.findall("joint"):
    if joint.get("name") in NAMES:
        limits = joint.find("limit")
        bounds[joint.get("name")] = np.rad2deg([float(limits.get("lower")), float(limits.get("upper"))])

def check(request):
    q = np.asarray(request["joints_deg"], dtype=float)
    delta = np.asarray(request["delta_m"], dtype=float)
    if q.shape != (5,) or delta.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(delta).all():
        raise ValueError("Expected five finite joint angles and three finite position deltas")
    if np.linalg.norm(delta) > 0.003:
        raise ValueError("Position delta exceeds 3 mm per check")
    for name, angle in zip(NAMES, q):
        low, high = bounds[name]
        if not low <= angle <= high:
            raise ValueError(f"{name} {angle:.2f} degrees outside model limits {low:.2f}..{high:.2f}")
    kin = module.RobotKinematics(str(MODEL), joint_names=NAMES)
    current = kin.forward_kinematics(q)
    target = current.copy(); target[:3,3] += delta
    solved = kin.inverse_kinematics(q, target)
    if not np.isfinite(solved).all():
        raise ValueError("IK returned non-finite values")
    max_change = float(np.max(np.abs(solved-q)))
    error = float(np.linalg.norm(kin.forward_kinematics(solved)[:3,3]-target[:3,3]))
    if max_change > 3 or error > .001:
        raise ValueError(f"Rejected IK: largest joint change {max_change:.2f} deg, position error {error*1000:.2f} mm")
    for name, angle in zip(NAMES, solved):
        low, high = bounds[name]
        if not low <= angle <= high:
            raise ValueError(f"IK output {name} outside model limits")
    return {"ok": True, "joints_deg": solved.tolist(), "max_change_deg": max_change, "error_mm": error*1000}

for line in sys.stdin:
    try:
        print(json.dumps(check(json.loads(line))), flush=True)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
