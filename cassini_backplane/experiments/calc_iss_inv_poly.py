from polymath import *
import numpy as np
import scipy.linalg as linalg
import oops
import oops.inst.cassini.iss as iss
import matplotlib.pyplot as plt
import cProfile, pstats, StringIO

def _invpoly_coeffs(u_axis_vals, v_axis_vals, order):
    """Internal routine for creating the coefficient matrix."""
    
    uvalues = u_axis_vals[:,np.newaxis]
    vvalues = v_axis_vals[np.newaxis,:]
    
    upowers = [1.]
    vpowers = [1.]

    nparams = (order+1) * (order+2) / 2
    a3d = np.empty((u_axis_vals.shape[0], v_axis_vals.shape[0], nparams))
    a3d[:,:,0] = 1.

    k = 0
    for p in range(1,order+1):
        upowers.append(upowers[-1] * uvalues)
        vpowers.append(vpowers[-1] * vvalues)

        for q in range(p+1):
            k += 1
            a3d[:,:,k] = vpowers[q] * upowers[p-q]

    return a3d

def invpoly_fit(data, u_axis_vals, v_axis_vals, order=2):
    nparams = (order+1) * (order+2) / 2

    a3d = _invpoly_coeffs(u_axis_vals, v_axis_vals, order)

    a2d  = a3d.reshape((data.size, nparams))
    b1d  = data.flatten()
        
    coeffts = linalg.lstsq(a2d, b1d)[0]
        
    return coeffts

def invpoly_compute(u_axis_vals, v_axis_vals, coeffs):
    order = int(np.sqrt(len(coeffs)*2))-1
    a3d = _invpoly_coeffs(u_axis_vals, v_axis_vals, order)
    result = np.sum(coeffs * a3d, axis=-1)
    return result

def poly_reshape(coeffs):
    order = int(np.sqrt(len(coeffs)*2))-1
    ret = np.zeros((order+1,order+1))
    ret[0,0] = coeffs[0]
    k = 0
    for p in range(1,order+1):
        for q in range(p+1):
            k += 1
            ret[p-q,q] = coeffs[k]
    return ret

def verify_iss(filename):
    obs = iss.from_file(filename, fast_distortion=False)
    
    fov = obs.fov
    
    print fov.xy_from_uv((500,500))
    print fov.xy_from_uv((512,512))
    print fov.uv_from_xy((0.,0.))
    
    meshgrid = oops.Meshgrid.for_fov(fov)
    
    # The Cassini distortion polynomial converts XY -> UV
    
    pr = cProfile.Profile()
    pr.enable()
    xy = fov.xy_from_uv(meshgrid.uv) # This solves the polynomial
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    ps.print_callers()
    print s.getvalue()

    print 'MESHGRID'
    print meshgrid.uv
    print 'XY'
    print xy
    uv = fov.uv_from_xy(xy)
    print 'NEW UV'
    print uv

    old_u, old_v = meshgrid.uv.to_scalars(recursive=False)
    old_u = old_u.vals
    old_v = old_v.vals
    new_u, new_v = uv.to_scalars(recursive=False)
    print
    print
    print new_u
    print
    print
    
    new_u = new_u.vals
    new_v = new_v.vals
        
    print old_u-new_u
    
    print 'U DIFF MIN MAX', np.min(old_u-new_u), np.max(old_u-new_u)
    print 'V DIFF MIN MAX', np.min(old_v-new_v), np.max(old_v-new_v)

NAC_FILENAME = 't:/external/cassini/derived/COISS_2xxx\COISS_2060/data/1644781751_1644850420/N1644783429_1_CALIB.IMG'
WAC_FILENAME = 'T:/external/cassini/derived/COISS_2xxx/COISS_2014/data/1498874330_1499156249/W1498874330_1_CALIB.IMG'


verify_iss(WAC_FILENAME)    
assert False

obs = iss.from_file(WAC_FILENAME)

fov = obs.fov
meshgrid = oops.Meshgrid.for_fov(fov, origin=(0.5,0.5), limit=(511.5,1023.5))
print fov.xy_from_uv(fov.uv_los+(-1,1))
#meshgrid = oops.Meshgrid.for_fov(fov)
print 'UV[0,0]', meshgrid.uv[0,0]
print 'UV[1,0]', meshgrid.uv[1,0]
print 'UV[0,1]', meshgrid.uv[0,1]
u_axis_vals = (meshgrid.uv-fov.uv_los).to_scalars()[0].vals[:,0]
v_axis_vals = (meshgrid.uv-fov.uv_los).to_scalars()[1].vals[0,:]

print 'U AXIS', u_axis_vals[0], u_axis_vals[-1]
print 'V AXIS', v_axis_vals[0], v_axis_vals[-1]

# The Cassini distortion polynomial converts XY -> UV
# We want the polynomial that converts UV -> XY
iss_xy = fov.xy_from_uv(meshgrid.uv) # This solves the polynomial
iss_x,iss_y = iss_xy.to_scalars(recursive=False)
iss_x = iss_x.vals
iss_y = iss_y.vals
print 'ISS X[0,0]', iss_x[0,0]
print 'ISS X[1,0]', iss_x[1,0]
print 'ISS X[0,1]', iss_x[0,1]
print 'ISS X', iss_x[0,0], iss_x[-1,-1], iss_x.shape
print 'ISS Y', iss_y[0,0], iss_y[-1,-1], iss_y.shape

x_coeffs = invpoly_fit(iss_x, u_axis_vals, v_axis_vals, order=3) # UV -> X
y_coeffs = invpoly_fit(iss_y, u_axis_vals, v_axis_vals, order=3) # UV -> Y

print poly_reshape(y_coeffs)[1,0]
print poly_reshape(y_coeffs)[0,1]

print 'X COEFFS'
print poly_reshape(x_coeffs)
print
print 'Y COEFFS'
print poly_reshape(y_coeffs)

new_x = invpoly_compute(u_axis_vals, v_axis_vals, x_coeffs)
new_y = invpoly_compute(u_axis_vals, v_axis_vals, y_coeffs)

print
print 'ISSX2', np.sum(iss_x**2), 'Y2', np.sum(iss_y**2)
print 'NEWX2', np.sum(new_x**2), 'NEWY2', np.sum(new_y**2)

print 'X DIFF MIN MAX', np.min(iss_x-new_x), np.max(iss_x-new_x)
print 'Y DIFF MIN MAX', np.min(iss_y-new_y), np.max(iss_y-new_y)

new_uv = fov.uv_from_xy(Pair.from_scalars(new_x, new_y))

new_u, new_v = new_uv.to_scalars()
new_u = new_u.vals
new_v = new_v.vals

old_u, old_v = meshgrid.uv.to_scalars()
old_u = old_u.vals
old_v = old_v.vals

print 'U DIFF MIN MAX', np.min(new_u-old_u), np.max(new_u-old_u)
print 'V DIFF MIN MAX', np.min(new_v-old_v), np.max(new_v-old_v)

#plt.plot((0,1023),(iss_x[0,0],iss_x[1023,0]), '-', color='red', lw=3, label='ISS X')
plt.plot(iss_x[:,0], '-', color='black', lw=1, label='ISSX[:,0]')
plt.plot(new_x[:,0], '--', color='black', lw=2, label='NEWX[:,0]')
plt.plot(iss_x[:,511], '-', color='cyan', lw=1, label='ISSX[:,511]')
plt.plot(new_x[:,511], '--', color='cyan', lw=2, label='NEWX[:,511]')
plt.plot(iss_y[0,:], '-', color='red', lw=1, label='ISSY[0,:]')
plt.plot(new_y[0,:], '--', color='red', lw=2, label='NEWY[0,:]')
plt.plot(iss_y[511,:], '-', color='green', lw=1, label='ISSY[511,:]')
plt.plot(new_y[511,:], '--', color='green', lw=2, label='NEWY[511,:]')
plt.legend()
plt.show()
