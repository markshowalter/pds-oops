import oops.inst.cassini.iss as iss
import oops

from cb_util_file import *

first_image_number = 1727769741
last_image_number  = 9999999999

for image_path in yield_image_filenames(
        first_image_number, last_image_number):
    obs = iss.from_file(image_path)
    if np.min(obs.data) > 0.5:
        print image_path.replace('t:\\external\\cassini\\derived\\COISS_2xxx\\', '')
#        print '- BAD'
#    print '- OK'