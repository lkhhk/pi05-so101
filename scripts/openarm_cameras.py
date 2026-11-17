"""Camera conventions shared by OpenArm collection and evaluation."""
import mujoco


def front_camera():
    # The source scene's named frontcam does not show the moving robot. This
    # fixed free camera uses the viewpoint from the upstream showcase renderer.
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.30, 0.0, 0.50]
    camera.distance = 0.95
    camera.azimuth = 150
    camera.elevation = -16
    return camera
