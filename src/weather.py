import math
def wind_to_enu(dir_deg, spd_kt):
    spd_ms = spd_kt*0.514444
    to_rad = math.radians((dir_deg+180)%360)
    return spd_ms*math.cos(to_rad), spd_ms*math.sin(to_rad)
