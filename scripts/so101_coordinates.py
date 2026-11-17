"""Preserve the existing IK wrist frame when the encoder homing offset changes."""


def wrist_zero(calibration):
    # Existing IK reference: raw 876 with Homing_Offset -413. This preserves that
    # geometric reference; it does not claim to recalibrate the URDF itself.
    return (876 - 413 - calibration['homing_offset']) % 4096
