"""Separate measured target limits from small encoder readback deviations."""
import math

READBACK_TOLERANCE_TICKS = 16


def check_readback(name, value, calibration):
    if not math.isfinite(value) or not 0 <= value <= 4095:
        raise ValueError(f'{name}: 编码器读数无效：{value}')
    low, high = calibration['range_min'], calibration['range_max']
    excess = max(low - value, value - high, 0)
    if excess > READBACK_TOLERANCE_TICKS:
        raise ValueError(f'{name}: 实际读数 {value}，校准范围 {low}..{high}，'
                         f'超出 {excess} ticks（回读容差 {READBACK_TOLERANCE_TICKS} ticks）')
