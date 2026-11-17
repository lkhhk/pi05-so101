"""SO-101 joint jog. Dry run by default; --move allows supervised motion."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "pygame"))
import pygame
import scservo_sdk as sdk

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAL = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/my_follower_arm.json"

def checked(result, what):
    if result[-2:] != (0, 0):
        raise RuntimeError(f"{what} failed: {result}")
    return result[0] if len(result) == 3 else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--move", action="store_true", help="enable motor commands")
    args = parser.parse_args()
    calibration = json.loads(CAL.read_text(encoding="utf-8"))
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() != 1:
        raise RuntimeError("Connect exactly one controller and retry.")
    j = pygame.joystick.Joystick(0); j.init()
    if j.get_numaxes() < 6 or j.get_numbuttons() < 5 or j.get_numhats() < 1:
        raise RuntimeError("Controller lacks expected Xbox inputs.")
    pygame.event.pump()
    if j.get_button(4):
        raise RuntimeError("Release LB before starting.")
    port = sdk.PortHandler("COM6"); port.baudrate = 1000000
    motor = sdk.PacketHandler(0)
    if not port.openPort():
        raise RuntimeError("Cannot open COM6.")
    active = set()
    targets = {}
    last_diagnostic = 0.0
    try:
        for name in NAMES:
            c = calibration[name]; mid = c["id"]
            vals = [checked(motor.read2ByteTxRx(port, mid, addr), f"{name} register {addr}") for addr in (9, 11, 31)]
            off = vals[2]; off = -(off & 2047) if off & 2048 else off
            if [vals[0], vals[1], off] != [c["range_min"], c["range_max"], c["homing_offset"]]:
                raise RuntimeError(f"Calibration mismatch: {name}; no motion sent.")
        print("Controller:", j.get_name(), "| mode:", "MOTION" if args.move else "DRY RUN")
        print("Hold LB to move. Left stick: base/shoulder; right stick: wrist roll/elbow.")
        print("D-pad up/down: wrist pitch; LT/RT: gripper. Ctrl+C exits.")
        print("Release LB to stop issuing targets. Enabled motors may hold; support arm before power off.")
        last = 0.0
        armed = False
        while True:
            events = pygame.event.get()
            if any(e.type == pygame.JOYDEVICEREMOVED and e.instance_id == j.get_instance_id() for e in events):
                break
            if pygame.joystick.get_count() != 1 or not j.get_init():
                break
            held = bool(j.get_button(4))
            axes = [j.get_axis(i) for i in range(6)]
            hat = j.get_hat(0)
            def direction(value):
                return 1 if value > 0.35 else -1 if value < -0.35 else 0
            # SDL Xbox mapping: axes 0/1 left stick, 2/3 right stick, 4/5 LT/RT.
            directions = [direction(axes[0]), direction(axes[1]), direction(axes[3]),
                          -hat[1], direction(axes[2]),
                          int(axes[5] > -0.3) - int(axes[4] > -0.3)]
            if not held:
                armed = False
            elif not any(directions):
                armed = True
            moving = set()
            if args.move and held and armed:
                for name, delta in zip(NAMES, directions):
                    if not delta:
                        continue
                    c = calibration[name]; mid = c["id"]
                    pos = checked(motor.read2ByteTxRx(port, mid, 56), f"read {name}")
                    if not c["range_min"] <= pos <= c["range_max"]:
                        raise RuntimeError(f"{name} position outside calibrated range.")
                    if mid not in active:
                        checked(motor.write2ByteTxRx(port, mid, 42, pos), "initialize goal")
                        checked(motor.write1ByteTxRx(port, mid, 40, 1), "enable torque")
                        targets[mid] = pos
                    targets[mid] = max(pos - 30, min(pos + 30, targets[mid] + delta * 2))
                    goal = max(c["range_min"], min(c["range_max"], targets[mid]))
                    checked(motor.write2ByteTxRx(port, mid, 42, goal), "write goal")
                    if time.monotonic() - last_diagnostic > 0.5:
                        actual_goal = checked(motor.read2ByteTxRx(port, mid, 42), "read goal")
                        torque = checked(motor.read1ByteTxRx(port, mid, 40), "read torque")
                        print(f"\nCOMMAND joint={name} pos={pos} goal_sent={goal} goal_read={actual_goal} torque={torque}", flush=True)
                        last_diagnostic = time.monotonic()
                    moving.add(mid)
            active = moving
            if time.monotonic() - last > 0.3:
                print(f"LB={int(held)} armed={int(armed)} directions={directions} axes={[round(x, 2) for x in axes]} hat={hat}    ", end="\r", flush=True)
                last = time.monotonic()
            time.sleep(0.05)
        print("\nController disconnected; stopped issuing movement commands.")
    finally:
        port.closePort()
        pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Motors already enabled may hold position; support arm before power off.")
    except Exception as exc:
        print(f"\nStopped: {exc}", file=sys.stderr)
        sys.exit(1)
