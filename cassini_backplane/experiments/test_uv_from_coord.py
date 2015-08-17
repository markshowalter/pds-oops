import oops
import polymath
import oops.inst.cassini.iss as iss

def bodies_latitude_longitude_to_pixels(obs, body_name, latitude, longitude,
                                        latlon_type='centric',
                                        lon_direction='east',
                                        underside=False):
    """Convert latitude,longitude pairs to U,V."""
    assert latlon_type in ('centric', 'graphic', 'squashed')
    assert lon_direction in ('east', 'west')

    latitude = polymath.Scalar.as_scalar(latitude)
    longitude = polymath.Scalar.as_scalar(longitude)
    
    if len(longitude) == 0:
        return np.array([]), np.array([])

    moon_surface = oops.Body.lookup(body_name).surface

    # Get the 'squashed' latitude    
    if latlon_type == 'centric':
        longitude = moon_surface.lon_from_centric(longitude)
        latitude = moon_surface.lat_from_centric(latitude, longitude)
    elif latlon_type == 'graphic':
        longitude = moon_surface.lon_from_graphic(longitude)
        latitude = moon_surface.lat_from_graphic(latitude, longitude)

    # Internal longitude is always 'east'
    if lon_direction == 'west':
        longitude = (-longitude) % oops.TWOPI

    uv = obs.uv_from_coords(moon_surface, (longitude, latitude), 
                            underside=underside)
    
    return uv

path = 't:/external/cassini/derived/COISS_2xxx/COISS_2009/data/1484688237_1484761461/N1484688342_1_CALIB.IMG'
latlon_type = 'centric'
lon_direction = 'east'
body_name = 'MIMAS'
coord_x = 353
coord_y = 869

obs = iss.from_file(path, fast_distortion=True)

bp = oops.Backplane(obs)

bp_latitude = bp.latitude(body_name, lat_type=latlon_type)

bp_longitude = bp.longitude(body_name, direction=lon_direction,
                            lon_type=latlon_type)
bp_emission = bp.emission_angle(body_name)

one_lat = bp_latitude[coord_y, coord_x].vals
one_lon = bp_longitude[coord_y, coord_x].vals
one_emission = bp_emission[coord_y, coord_x]

print 'XY', coord_x, coord_y
print 'LATLON', one_lat, one_lon
print 'E', one_emission

uv = bodies_latitude_longitude_to_pixels(obs, body_name, [one_lat], [one_lon],
                                        latlon_type=latlon_type,
                                        lon_direction=lon_direction,
                                        underside=True)
print 'UNDERSIDE', uv

uv = bodies_latitude_longitude_to_pixels(obs, body_name, [one_lat], [one_lon],
                                        latlon_type=latlon_type,
                                        lon_direction=lon_direction,
                                        underside=False)
print 'NOT UNDERSIDE', uv
