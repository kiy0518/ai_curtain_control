"""Sunrise/sunset times (pure Python, no dependency).

Standard "sunrise equation" — accurate to ~1-2 minutes, plenty for curtains.
Returns local minutes-of-day for sunrise/sunset, or None at polar day/night.
"""

import datetime
import math

_ZENITH = 90.833   # official sunrise/sunset (incl. atmospheric refraction)


def _event(lat, lon, y, mo, d, tz_off_sec, is_sunrise):
    N = (datetime.date(y, mo, d).toordinal()
         - datetime.date(y, 1, 1).toordinal() + 1)
    lng_hour = lon / 15.0
    t = N + ((6 if is_sunrise else 18) - lng_hour) / 24.0
    M = 0.9856 * t - 3.289
    L = (M + 1.916 * math.sin(math.radians(M))
         + 0.020 * math.sin(math.radians(2 * M)) + 282.634) % 360
    RA = math.degrees(math.atan(0.91764 * math.tan(math.radians(L)))) % 360
    RA = (RA + ((math.floor(L / 90) * 90) - (math.floor(RA / 90) * 90))) / 15.0
    sin_dec = 0.39782 * math.sin(math.radians(L))
    cos_dec = math.cos(math.asin(sin_dec))
    cos_h = ((math.cos(math.radians(_ZENITH)) - sin_dec * math.sin(math.radians(lat)))
             / (cos_dec * math.cos(math.radians(lat))))
    if cos_h > 1 or cos_h < -1:
        return None
    H = (360 - math.degrees(math.acos(cos_h))) if is_sunrise \
        else math.degrees(math.acos(cos_h))
    H /= 15.0
    T = H + RA - 0.06571 * t - 6.622
    ut = (T - lng_hour) % 24
    local = (ut + tz_off_sec / 3600.0) % 24
    return local * 60.0   # minutes of day


def sun_times(lat, lon, y, mo, d, tz_off_sec):
    """Return (sunrise_min, sunset_min) local minutes-of-day (or None each)."""
    return (_event(lat, lon, y, mo, d, tz_off_sec, True),
            _event(lat, lon, y, mo, d, tz_off_sec, False))
