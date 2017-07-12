import sys
import os.path
import fring_util
from optparse import OptionParser

cmd_line = sys.argv[1:]

if len(cmd_line) == 0:
    cmd_line = ['-a']

parser = OptionParser()

fring_util.add_parser_options(parser)

options, args = parser.parse_args(cmd_line)

obsid_db = {}

for obsid, image_name, image_path in fring_util.enumerate_files(options, args, '_CALIB.IMG'):
    if not obsid in obsid_db:
        obsid_db[obsid] = []
    obsid_db[obsid].append(image_name)

print 'FRING_FILENAMES = {'

for obsid in sorted(obsid_db.keys()):
    obsid_db[obsid].sort(key=lambda x: x[1:13]+x[0])
    min_num = int(obsid_db[obsid][0][1:11])
    max_num = int(obsid_db[obsid][-1][1:11])
    print "    '" + obsid + "': (" + str(min_num) + ', ' + str(max_num) + ', ['
    for filename in obsid_db[obsid]:
        print "        '" + filename + "',"
    print "        ]),"
print '}'

