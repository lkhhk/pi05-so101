"""Show connected gamepads and their inputs; never connects to the robot."""
import os
import sys
import time
from pathlib import Path
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "pygame"))
import pygame

pygame.init()
pygame.joystick.init()
try:
    if not pygame.joystick.get_count():
        print("No gamepad found. Connect and power on the Xbox controller, then retry.")
        raise SystemExit(1)
    for i in range(pygame.joystick.get_count()):
        j = pygame.joystick.Joystick(i)
        j.init()
        print(f"{i}: {j.get_name()} | axes={j.get_numaxes()} buttons={j.get_numbuttons()} hats={j.get_numhats()}")
    j = pygame.joystick.Joystick(0)
    print("Move sticks and press buttons; Ctrl+C to exit.")
    while pygame.joystick.get_count() > 0 and j.get_init():
        pygame.event.pump()
        axes = [round(j.get_axis(i), 2) for i in range(j.get_numaxes())]
        buttons = [i for i in range(j.get_numbuttons()) if j.get_button(i)]
        hats = [j.get_hat(i) for i in range(j.get_numhats())]
        print(f"axes={axes} pressed={buttons} hats={hats}       ", end="\r", flush=True)
        time.sleep(0.1)
    print("\nController disconnected.")
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    pygame.quit()
