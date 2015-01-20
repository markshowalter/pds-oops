import numpy as np
import oops
import oops.inst.cassini.iss as iss
import polymath

LAT_TYPE = 'graphic'
LONG_DIRECTION = 'east'

def bodies_latitude_longitude_to_pixels(obs, body_name, latitude, longitude,
                                        lat_type='centric',
                                        direction='east'):
    """Convert latitude,longitude pairs to U,V."""
    assert lat_type in ('centric', 'graphic', 'squashed')
    assert direction in ('east', 'west')

    latitude = polymath.Scalar.as_scalar(latitude)
    longitude = polymath.Scalar.as_scalar(longitude)
    
    if len(longitude) == 0:
        return np.array([]), np.array([])

    moon_surface = oops.Body.lookup(body_name).surface

    # Internal longitude is always 'east'
    if direction == 'west':
        longitude = (-longitude) % oops.TWOPI

    # Get the 'squashed' latitude    
    if lat_type == 'centric':
        longitude = moon_surface.lon_from_centric(longitude)
        latitude = moon_surface.lat_from_centric(latitude, longitude)
    elif lat_type == 'graphic':
        longitude = moon_surface.lon_from_graphic(longitude)
        latitude = moon_surface.lat_from_graphic(latitude, longitude)

        
    obs_event = oops.Event(obs.midtime, (polymath.Vector3.ZERO,
                                         polymath.Vector3.ZERO),
                           obs.path, obs.frame)
    _, obs_event = moon_surface.photon_to_event_by_coords(
                                          obs_event, (longitude, latitude))

    uv = obs.fov.uv_from_los(-obs_event.arr)
    u, v = uv.to_scalars()
    
    return u.vals, v.vals
    
def process_moon_one_file(filename, body_name):
    obs = iss.from_file(filename)

    # Given the offset, create a new offset FOV and compute the image
    # metadata
    surface = oops.Body.lookup(body_name).surface
#    oops.Body.lookup(body_name).surface = oops.surface.Spheroid(surface.origin, surface.frame,
#                                                        (surface.radii[0], surface.radii[2]))

    offset_u = -14
    offset_v = -24
    
    orig_fov = obs.fov 
    obs.fov = oops.fov.OffsetFOV(obs.fov, (offset_u, offset_v))
    bp = oops.Backplane(obs)

    bp_latitude = bp.latitude(body_name, lat_type=LAT_TYPE)
    bp_latitude = bp_latitude.vals.astype('float')

    bp_longitude = bp.longitude(body_name, direction=LONG_DIRECTION, lon_type=LAT_TYPE)
    bp_longitude = bp_longitude.vals.astype('float')

    obs.fov = orig_fov
    
    # Test invertability
    for u in range(0,730,10):
        for v in range(140,670,10):
            lat = bp_latitude[v,u]
            lon = bp_longitude[v,u]
            u_pixels, v_pixels = bodies_latitude_longitude_to_pixels(obs, body_name, [lat], [lon],
                                                                     lat_type=LAT_TYPE,
                                                                     direction=LONG_DIRECTION)
            print '%4d %4d %6.2f %6.2f => %7.4f %7.4f' % (
                      u,v,lat*oops.DPR,lon*oops.DPR,
                      u_pixels[0]-u+offset_u-.5, v_pixels[0]-v+offset_v-.5)
            
#filename = r'T:\external\cassini\derived\COISS_2xxx\COISS_2008\data\1484506648_1484573247/N1484530421_1_CALIB.IMG'
filename = r't:\external\cassini\derived\COISS_2xxx\COISS_2014\data\1501618408_1501647096\N1501645855_1_CALIB.IMG'

process_moon_one_file(filename, 'MIMAS')
