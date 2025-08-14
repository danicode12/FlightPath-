from pyproj import Transformer
def latlonalt_to_enu(lat, lon, alt, lat0, lon0, alt0):
    t_ecef = Transformer.from_crs("epsg:4979","epsg:4978",always_xy=True)
    x,y,z   = t_ecef.transform(lon,lat,alt)
    x0,y0,z0= t_ecef.transform(lon0,lat0,alt0)
    t_enu = Transformer.from_crs(
        {"proj":"geocent","ellps":"WGS84","datum":"WGS84"},
        {"proj":"enu","lat_0":lat0,"lon_0":lon0,"h_0":alt0,"datum":"WGS84","ellps":"WGS84"},
        always_xy=True
    )
    E,N,U = t_enu.transform(x,y,z,x0=x0,y0=y0,z0=z0)
    return E,N,U
