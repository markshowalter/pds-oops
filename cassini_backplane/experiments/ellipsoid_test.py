import numpy as np
import oops
import oops.inst.cassini.iss as iss
from polymath import *


def moons_latitude_longitude_to_pixels(obs, body_name, latitude, longitude):
    latitude = np.asarray(latitude)
    longitude = np.asarray(longitude)
    
    if len(longitude) == 0:
        return np.array([]), np.array([])
    
    moon_surface = oops.Body.lookup(body_name).surface
    obs_event = oops.Event(obs.midtime, (Vector3.ZERO,Vector3.ZERO),
                           obs.path, obs.frame)
    _, obs_event = moon_surface.photon_to_event_by_coords(obs_event,
                                      (longitude*oops.RPD, latitude*oops.RPD))

    uv = obs.fov.uv_from_los(-obs_event.arr)
    u, v = uv.to_scalars()
    
    return u.vals, v.vals
    
def process_moon_one_file(filename, body_name):
    obs = iss.from_file(filename)

    # Given the offset, create a new offset FOV and compute the image
    # metadata
    surface = oops.Body.lookup(body_name).surface
    oops.Body.lookup(body_name).surface = oops.surface.Spheroid(surface.origin, surface.frame,
                                                        (surface.radii[0], surface.radii[2]))

    obs.fov = oops.fov.OffsetFOV(obs.fov, (-32, 5))
    bp = oops.Backplane(obs)

    bp_latitude = bp.latitude(body_name, lat_type='centric')
    bp_latitude = bp_latitude.vals.astype('float') * oops.DPR

    bp_longitude = bp.longitude(body_name, direction='east')
    bp_longitude = bp_longitude.vals.astype('float') * oops.DPR

    # Test invertability
    for u in range(532,710,20):
        for v in range(518,686,20):
            lat = bp_latitude[v,u]
            lon = bp_longitude[v,u]
            u_pixels, v_pixels = moons_latitude_longitude_to_pixels(obs, body_name, [lat], [lon])
            print u, 'VS', u_pixels[0], '/', v, 'VS', v_pixels[0]

filename = 'j:/Temp/N1484530421_1_CALIB.IMG'

process_moon_one_file(filename, 'MIMAS')
