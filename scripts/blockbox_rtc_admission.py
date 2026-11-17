"""Latency admission for the asynchronous 50-step guided RTC deployment."""
import math
from collections import deque


def admitted(latency_s,refill=35):
    delay=math.ceil(latency_s*30)+2
    return delay*2+4<50 and delay+2<refill


def preflight(sample,report,refill=35,on_retry=lambda *_:None):
    """Require five consecutive admissible samples; bound recovery to 15 probes.

    A slow startup sample is not sufficient evidence of a persistently slow
    route. It may age out only after five newer guided samples meet the same
    original latency bound. Timeouts and protocol errors are never swallowed.
    """
    result,elapsed=sample(None)
    report(result,elapsed)
    recent=deque(maxlen=5)
    for attempt in range(1,16):
        result,elapsed=sample(result['normalized_actions'][:35])
        report(result,elapsed)
        recent.append(elapsed)
        if len(recent)==5:
            if admitted(max(recent),refill):return list(recent)
            on_retry(attempt,max(recent))
    max_delay_budget=min((50-5)//2,refill-3)
    limit_s=max(0,(max_delay_budget-2)/30)
    raise RuntimeError(
        f'RTC latency admission failed after 15 guided probes: recent maximum '
        f'{max(recent):.3f}s exceeds the {limit_s:.3f}s admission budget at 30 Hz. '
        'No motion started; check network traffic or retry when the route is stable.'
    )
