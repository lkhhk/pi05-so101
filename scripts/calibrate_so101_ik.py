"""Interactive LeRobot calibration; requires a person supporting the arm."""
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "vendor" / "lerobot043"), str(root / "vendor" / "pygame")]
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

config = SO101FollowerConfig(port="COM6", id="my_follower_ik")
robot = SO101Follower(config)
if robot.calibration_fpath.exists():
    raise SystemExit(f"Calibration file already exists: {robot.calibration_fpath}. Stopping to preserve it.")
print("This runs official LeRobot calibration on COM6 and writes new settings to the motors.")
print("Support the whole arm: calibration will disable motor torque.")
print("The existing my_follower_arm file remains saved, but old control scripts may no longer match the motors.")
if input("Type START to continue, or press Enter to cancel: ").strip() != "START":
    raise SystemExit("Cancelled; no motor settings changed.")
try:
    robot.bus.connect()
    robot.calibrate()
    print(f"New calibration: {robot.calibration_fpath}")
finally:
    if robot.bus.is_connected:
        robot.bus.disconnect(disable_torque=False)
