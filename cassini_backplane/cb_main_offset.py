###############################################################################
# cb_main_offset.py
#
# The main top-level driver for single file offset finding.
###############################################################################

from cb_logging import *
import logging

import argparse
import cProfile, pstats, StringIO
import os
import subprocess
import sys
import time
import traceback

import oops.inst.cassini.iss as iss
import oops

from cb_config import *
from cb_gui_offset_data import *
from cb_offset import *
from cb_util_file import *


command_list = sys.argv[1:]

if len(command_list) == 0:
#    command_line_str = '--has-png-file --force-offset'
#     command_line_str = '--first-image-num 1487299402 --last-image-num 1487302209 --max-subprocesses 4'
#     command_line_str = '--first-image-num 1481738274 --last-image-num 1496491595 --force-offset --image-console-level none --max-subprocesses 4'
#     command_line_str = '--first-image-num 1637518901 --last-image-num 1665998079 --image-console-level none --max-subprocesses 4'
#N1736967486_1
#N1736967706_1
#    command_line_str = '''--force-offset --image-console-level debug --display-offset-results
#N1760870348_1'''
#    command_line_str = '--force-offset N1496877261_8 --image-console-level debug --profile'
#    command_line_str = '--image-pds-csv t:/external/cb_support/titan-clear-151203.csv --stars-only --max-subprocesses 4'

#    command_line_str = 'N1595336241_1 --force-offset --image-console-level debug --display-offset-results' # Star smear with edge of A ring
#    command_line_str = 'N1751425716_1 --force-offset --image-console-level debug --display-offset-results' # Star smear with nothing else
    command_line_str = 'N1484580522_1 --force-offset --image-console-level debug --display-offset-results --stars-label-font courbd.ttf,30 --png-label-font courbd.ttf,30' # Star smear with Mimas

#    command_line_str = 'N1654250545_1 --force-offset --image-console-level debug --display-offset-results' # Rings closeup - A ring - no features
#    command_line_str = 'N1477599121_1 --force-offset --image-console-level debug --display-offset-results --rings-label-font courbd.ttf,30' # Colombo->Huygens closeup
#    command_line_str = 'N1588310978_1 --force-offset --image-console-level debug --display-offset-results' # Colombo->Huygens closeup
#    command_line_str = 'N1600327271_1 --force-offset --image-console-level debug --display-offset-results' # Colombo->Huygens closeup
#    command_line_str = 'N1608902918_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Colombo->Huygens closeup
#    command_line_str = 'N1624548280_1 --force-offset --image-console-level debug --display-offset-results' # Colombo->Huygens closeup
#    command_line_str = 'W1532487683_1 --force-offset --image-console-level debug --display-offset-results' # Wide angle ring with full B ring gap

#    command_line_str = 'N1589083632_1 --force-offset --image-console-level debug --display-offset-results' # A ring edge - no curvature
#    command_line_str = 'N1591063671_1 --force-offset --image-console-level debug --display-offset-results' # A ring edge - no curvature
#    command_line_str = 'N1595336241_1 --force-offset --image-console-level debug --display-offset-results' # A ring edge - no curvature
#    command_line_str = 'N1601009125_1 --force-offset --image-console-level debug --display-offset-results' # A ring edge - blur, low features, and stars
#    command_line_str = 'N1625958009_1 --force-offset --image-console-level debug --display-offset-results' # A ring edge - blur and stars
#    command_line_str = 'N1492060009_1 --force-offset --image-console-level debug --display-offset-results' # A ring edge - matches with limited features, stars
#    command_line_str = 'N1492072293_1 --force-offset --image-console-level debug --display-offset-results' # A ring edge - matches with limited features, stars
#    command_line_str = 'N1493613276_1 --force-offset --image-console-level debug --display-offset-results' # A ring edge - too low res, but enough other features
#    command_line_str = 'N1543168726_1 --force-offset --image-console-level debug --display-offset-results' # A ring edge - star and model match
#    command_line_str = 'N1601009320_1 --force-offset --image-console-level debug --display-offset-results' # High res A ring edge - only works with blurring - tests A ring special case for PNG - low confidence
#    command_line_str = 'N1595336719_1 --force-offset --image-console-level debug --display-offset-results' # Star streaks through the rings but stars in wrong place
#    command_line_str = 'N1755729895_1 --force-offset --image-console-level debug --display-offset-results' # A ring edge with circular A ring model but not used due to blurring
#    command_line_str = 'W1466448054_1 --force-offset --image-console-level debug --display-offset-results' # Distant WAC ring

#    command_line_str = 'N1495327885_1 --force-offset --image-console-level debug --display-offset-results' # Closeup with multiple gaps and ringlets
#    command_line_str = 'N1498373872_1 --force-offset --image-console-level debug --display-offset-results' # B ring edge too low res but other features OK
#    command_line_str = 'N1541685510_1 --force-offset --image-console-level debug --display-offset-results' # B ring edge, bad curvature
#    command_line_str = 'N1588249321_1 --force-offset --image-console-level debug --display-offset-results' # B ring edge too low res but other features OK
#    command_line_str = 'N1627296827_1 --force-offset --image-console-level debug --display-offset-results' # B ring edge too low res, other features out of frame
#    command_line_str = 'N1627301821_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # B ring edge but nothing in frame
#    command_line_str = 'N1630088199_1 --force-offset --image-console-level debug --display-offset-results' # B ring edge, bad curvature, Saturn behind

#    command_line_str = 'N1512448422_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Rhea and Dione overlapping
#    command_line_str = 'N1511716650_2 --force-offset --image-console-level debug --display-offset-results' # Rhea closeup but not whole image
#    command_line_str = 'N1511728708_2 --force-offset --image-console-level debug --display-offset-results' # Rhea whole image
    
#    command_line_str = 'W1515969272_1 --force-offset --image-console-level debug --display-offset-results' # Titan a=35 CLEAR
#    command_line_str = 'N1635365617_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan phase 109 CLEAR
#    command_line_str = 'N1527933271_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan phase 163 CLEAR
    
#    command_line_str = 'W1561880145_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=12 VIO
#    command_line_str = 'W1655808265_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=32 VIO
#    command_line_str = 'W1683615178_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=45 VIO
#    command_line_str = 'W1552216646_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=60 VIO
#    command_line_str = 'W1748064618_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=80 VIO
#    command_line_str = 'W1753508727_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=90 VIO
#    command_line_str = 'W1760465018_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=110 VIO
#    command_line_str = 'W1622977122_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=130 VIO
#    command_line_str = 'W1717682790_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=155 VIO

#    command_line_str = 'N1624600010_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=12 RED
#    command_line_str = 'N1580288525_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=18 RED with Saturn
#    command_line_str = 'N1624879028_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=81 RED
#    command_line_str = 'N1611563684_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=125 RED
#    command_line_str = 'N1614174403_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=156 RED

#    command_line_str = 'W1537751816_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=65 CLEAR off edge
#    command_line_str = 'N1702247210_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=109, Rhea overlapping BL1 (RHEA 29,-3) (Titan 29,-5)
#    command_line_str = 'N1686939958_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=100, Rhea (Rhea -10,-2) (Titan -10,0)
#    command_line_str = 'N1517095065_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=116 CLEAR (Titan 2,9)
#    command_line_str = 'W1578213940_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=126 CLEAR off edge 256x256
#    command_line_str = 'W1629878033_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=126 CLEAR off edge
#    command_line_str = 'N1806009125_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=141, Rhea, Mimas BL1

#    command_line_str = 'N1626123721_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=61, MT3 NAC version
#    command_line_str = 'W1536376241_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=61, MT3 WAC version 512x12

#    command_line_str = 'N1686939958_1 --force-offset --image-console-level debug --display-offset-results' # Titan Stars -10,-2

#    command_line_str = 'N1702239215_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=107 CLEAR + TETHYS (TETHYS NOT VIS) (TITAN 13,-5)
#    command_line_str = 'N1702240651_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=107 CLEAR + TETHYS (TETHYS NOT VIS) (TITAN 14,-3)
#    command_line_str = 'N1637959164_1 --force-offset --image-console-level debug --display-offset-results ' # Titan a=105 CLEAR + TETHYS (TETHYS NOT VIS) (TITAN 0,5)
#    command_line_str = 'N1635365691_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # Titan a=109 CLEAR + RHEA?? 

#    command_line_str = 'W1506215893_1 N1506215893_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # BOTSIM 
#    command_line_str = 'W1716174363_1 N1716174363_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # BOTSIM
#    command_line_str = 'W1613101946_1 N1613101946_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # BOTSIM
#    command_line_str = 'W1477437523_2 N1477437523_2 --force-offset --image-console-level debug --display-offset-results --no-allow-stars' # BOTSIM
#    command_line_str = 'W1610355001_1 N1610355001_1 --force-offset --image-console-level debug --no-allow-stars' # BOTSIM

#    command_line_str = 'N1656930617_1 --force-offset --image-console-level debug --display-offset-results --overlay-show-star-streaks' # Long streaks through the D ring, photometry OK without ring subtraction 
#    command_line_str = 'N1659250659_1 --force-offset --image-console-level debug --display-offset-results --overlay-show-star-streaks' # Long streaks, fails photometry 
#    command_line_str = 'N1575647907_1 --force-offset --image-console-level debug --display-offset-results --overlay-show-star-streaks' # Long streaks, requires pred kernel 
#    command_line_str = 'W1553000979_1 --force-offset --image-console-level debug --display-offset-results --overlay-show-star-streaks' # Long streaks, BL1, photometry OK
#    command_line_str = 'W1620385882_1 --force-offset --image-console-level debug --display-offset-results --overlay-show-star-streaks' # Medium streaks, IR2, photometry OK
#    command_line_str = 'N1498395457_1 --force-offset --image-console-level debug --display-offset-results --overlay-show-star-streaks' # Medium streaks, IR2, photometry OK


# MOSAIC TEST
#    command_line_str = 'N1597187315_1 --force-offset --image-console-level debug --display-offset-results --no-allow-stars --body-cartographic-data ENCELADUS=T:/cdaps-results/mosaics/ENCELADUS/ENCELADUS_0.00_-30.00_F_F_BL1_0001'

# BOOTSTRAPPING - ENCELADUS

#    command_line_str = '--force-offset --first-image-num 1487299402 --last-image-num 1487302209 --image-logfile-level debug --max-subprocesses 2'
#    command_line_str = '--force-offset --first-image-num 1516151439 --last-image-num 1516171418 --image-logfile-level debug --max-subprocesses 2'
#    command_line_str = '--force-offset --first-image-num 1597175743 --last-image-num 1597239766 --image-logfile-level debug --max-subprocesses 2'
#    command_line_str = '--force-offset --first-image-num 1602263870 --last-image-num 1602294337 --image-logfile-level debug --max-subprocesses 2'
#    command_line_str = '--force-offset --first-image-num 1604136974 --last-image-num 1604218747 --image-logfile-level debug --max-subprocesses 2'
#    command_line_str = '--force-offset --first-image-num 1637450504 --last-image-num 1637482321 --image-logfile-level debug --max-subprocesses 2'
#    command_line_str = '--force-offset --first-image-num 1652858990 --last-image-num 1652867484 --image-logfile-level debug --max-subprocesses 2'
#    command_line_str = '--force-offset --first-image-num 1660419699 --last-image-num 1660446193 --image-logfile-level debug --max-subprocesses 2'
#    command_line_str = '--force-offset --first-image-num 1669795989 --last-image-num 1669856551 --image-logfile-level debug --max-subprocesses 2'
#    command_line_str = '--force-offset --first-image-num 1671569397 --last-image-num 1671602206 --image-logfile-level debug --max-subprocesses 2'
#    command_line_str = '--force-offset --first-image-num 1694646860 --last-image-num 1694652019 --image-logfile-level debug --max-subprocesses 2'
#    command_line_str = '--force-offset --first-image-num 1697700931 --last-image-num 1697717648 --image-logfile-level debug --max-subprocesses 2'
#    command_line_str = '--force-offset --first-image-num 1702359393 --last-image-num 1702361420 --image-logfile-level debug --max-subprocesses 2'

# BOOTSRAPPING - MIMAS

#    command_line_str = '--force-offset --first-image-num 1501627117 --last-image-num 1501651303 --image-logfile-level debug --max-subprocesses 2'
#    command_line_str = '--force-offset --first-image-num 1521584495 --last-image-num 1521620702 --image-logfile-level debug --max-subprocesses 2'
#    command_line_str = '--force-offset --first-image-num 1644777693 --last-image-num 1644802455 --image-logfile-level debug --max-subprocesses 2'
#    command_line_str = '--force-offset --first-image-num 1717565987 --last-image-num 1717571685 --image-logfile-level debug --max-subprocesses 2'

#    command_line_str = 'N1511700120_1 W1511714612_1 N1511715316_2 N1511716650_2 N1511718003_2 N1511719101_2 W1511726740_1 --image-console-level debug --force-offset --no-allow-stars'
#    command_line_str = 'N1511726828_2 N1511726954_2 N1511727079_2 N1511727217_2 N1511727361_2 N1511727503_2 N1511727641_2 N1511727774_2 N1511727899_2 N1511728035_2 N1511728175_2 N1511728315_2 N1511728440_2 N1511728581_2 N1511728708_2 N1511728833_2 N1511728958_2 N1511729097_2 N1511729338_2 N1511729463_2 N1511729588_2 --image-console-level debug --force-offset'
#    command_line_str = 'W1511726740_1 --image-console-level debug --force-offset --no-allow-stars --display-offset-results'
#    command_line_str = 'N1511727641_2 --image-console-level debug --force-offset --no-allow-stars --display-offset-results'
#    command_line_str = 'N1511727503_2 --image-console-level debug --force-offset --no-allow-stars --display-offset-results'
#    command_line_str = 'N1511727079_2 --image-console-level debug --force-offset --no-allow-stars --display-offset-results'

#     command_line_str = '--force-offset --image-console-level debug --body-cartographic-data RHEA=t:/cdaps-results/mosaics/RHEA/RHEA__0.500_0.500_centric_east__180.00_-30.00_T_T_ALL_0003-MOSAIC.dat --image-full-path t:/external/cassini/derived/COISS_2xxx/COISS_2017/data/1511715235_1511729063/N1511716650_2_CALIB.IMG --no-allow-stars'
#    command_line_str = 'N1492180176_1 --image-console-level debug --force-offset --no-allow-stars --display-offset-results --bodies-label-font courbd.ttf,30'
    
    command_list = command_line_str.split()

parser = argparse.ArgumentParser(
    description='Cassini Backplane Main Interface for Offsets',
    epilog='''Default behavior is to perform an offset pass on all images
              without associated offset files''')

###XXXX####
# --image-logfile is incompatible with --max-subprocesses > 0

###############################
### Arguments about logging ###
###############################
parser.add_argument(
    '--main-logfile', metavar='FILENAME',
    help='''The full path of the logfile to write for the main loop; defaults 
            to $(CB_RESULTS_ROOT)/logs/cb_main_offset/<datetime>.log''')
LOGGING_LEVEL_CHOICES = ['debug', 'info', 'warning', 'error', 'critical', 'none']
parser.add_argument(
    '--main-logfile-level', metavar='LEVEL', default='info', 
    choices=LOGGING_LEVEL_CHOICES,
    help='Choose the logging level to be output to the main loop logfile')
parser.add_argument(
    '--main-console-level', metavar='LEVEL', default='info',
    choices=LOGGING_LEVEL_CHOICES,
    help='Choose the logging level to be output to stdout for the main loop')
parser.add_argument(
    '--image-logfile', metavar='FILENAME',
    help='''The full path of the logfile to write for each image file; 
            defaults to 
            $(CB_RESULTS_ROOT)/logs/<image-path>/<image_filename>-OFFSET-<datetime>.log''')
parser.add_argument(
    '--image-logfile-level', metavar='LEVEL', default='info',
    choices=LOGGING_LEVEL_CHOICES,
    help='Choose the logging level to be output to stdout for each image')
parser.add_argument(
    '--image-console-level', metavar='LEVEL', default='none',
    choices=LOGGING_LEVEL_CHOICES,
    help='Choose the logging level to be output to stdout for each image')
parser.add_argument(
    '--profile', action='store_true', 
    help='Do performance profiling')

####################################
### Arguments about subprocesses ###
####################################
parser.add_argument(
    '--is-subprocess', action='store_true',
    help='Internal flag used to indicate this process was spawned by a parent')
parser.add_argument(
    '--max-subprocesses', type=int, default=0, metavar='NUM',
    help='The maximum number jobs to perform in parallel')

##########################################
### Arguments about the offset process ###
##########################################
parser.add_argument(
    '--force-offset', action='store_true', default=False,
    help='Force offset computation even if the offset file exists')
parser.add_argument(
    '--offset-redo-error', action='store_true', default=False,
    help='''Force offset computation if the offset file exists and 
            indicates a fatal error''')
parser.add_argument(
    '--display-offset-results', action='store_true', default=False,
    help='Graphically display the results of the offset process')
parser.add_argument(
    '--botsim-offset', type=str, metavar='U,V',
    help='Force the offset to be u,v')
parser.add_argument(
    '--stars-only', action='store_true', default=False,
    help='Navigate only using stars')
parser.add_argument(
    '--allow-stars', dest='allow_stars', action='store_true', default=True,
    help='Include stars in navigation')
parser.add_argument(
    '--no-allow-stars', dest='allow_stars', action='store_false',
    help='Do not include stars in navigation')
parser.add_argument(
    '--rings-only', action='store_true', default=False,
    help='Navigate only using rings')
parser.add_argument(
    '--allow-rings', dest='allow_rings', action='store_true', default=True,
    help='Include rings in navigation')
parser.add_argument(
    '--no-allow-rings', dest='allow_rings', action='store_false',
    help='Do not include rings in navigation')
parser.add_argument(
    '--moons-only', action='store_true', default=False,
    help='Navigate only using moons')
parser.add_argument(
    '--allow-moons', dest='allow_moons', action='store_true', default=True,
    help='Include moons in navigation')
parser.add_argument(
    '--no-allow-moons', dest='allow_moons', action='store_false',
    help='Do not include moons in navigation')
parser.add_argument(
    '--saturn-only', action='store_true', default=False,
    help='Navigate only using Saturn')
parser.add_argument(
    '--allow-saturn', dest='allow_saturn', action='store_true', default=True,
    help='Include saturn in navigation')
parser.add_argument(
    '--no-allow-saturn', dest='allow_saturn', action='store_false',
    help='Do not include saturn in navigation')
parser.add_argument(
    '--body-cartographic-data', dest='body_cartographic_data', action='append',
    metavar='BODY=MOSAIC',
    help='The mosaic providing cartographic data for the given BODY')
parser.add_argument(
    '--use-predicted-kernels', action='store_true', default=False,
    help='Use predicted CK kernels')

#######################################################
### Arguments about overlay and PNG file generation ###
#######################################################
parser.add_argument(
    '--overlay-show-star-streaks', action='store_true', default=False,
    help='Show star streaks in the overlay and PNG files')
parser.add_argument(
    '--png-blackpoint', type=float, default=None,
    help='Set the blackpoint for the PNG file')
parser.add_argument(
    '--png-whitepoint', type=float, default=None,
    help='Set the whitepoint for the PNG file')
parser.add_argument(
    '--png-gamma', type=float, default=None,
    help='Set the gamma for the PNG file')
parser.add_argument(
    '--png-label-font', type=str, default=None, metavar='FONTFILE,SIZE',
    help='Set the font for the PNG metadata info')
parser.add_argument(
    '--stars-label-font', type=str, default=None, metavar='FONTFILE,SIZE',
    help='Set the font for star labels')
parser.add_argument(
    '--rings-label-font', type=str, default=None, metavar='FONTFILE,SIZE',
    help='Set the font for ring labels')
parser.add_argument(
    '--bodies-label-font', type=str, default=None, metavar='FONTFILE,SIZE',
    help='Set the font for body labels')

file_add_selection_arguments(parser)

arguments = parser.parse_args(command_list)


###############################################################################
#
# SUBPROCESS HANDLING
#
###############################################################################

def collect_cmd_line(image_path):
    ret = []
    ret += ['--is-subprocess']
    ret += ['--main-logfile-level', 'none']
    ret += ['--main-console-level', 'none']
    ret += ['--image-logfile-level', arguments.image_logfile_level]
    ret += ['--image-console-level', arguments.image_console_level]
    ret += ['--force-offset']
    if arguments.profile:
        ret += ['--profile']
    if not arguments.allow_stars:
        ret += ['--no-allow-stars']
    if not arguments.allow_rings:
        ret += ['--no-allow-rings']
    if not arguments.allow_moons:
        ret += ['--no-allow-moons']
    if not arguments.allow_saturn:
        ret += ['--no-allow-saturn']
    if arguments.use_predicted_kernels:
        ret += ['--use-predicted-kernels']
    if arguments.overlay_show_star_streaks:
        ret += ['--overlay-show-star-streaks']
    ret += ['--image-full-path', image_path]
    
    return ret

SUBPROCESS_LIST = []

def run_and_maybe_wait(args, image_path, bootstrapped):
    said_waiting = False
    while len(SUBPROCESS_LIST) == arguments.max_subprocesses:
        if not said_waiting:
            main_logger.debug('Waiting for a free subprocess')
            said_waiting = True
        for i in xrange(len(SUBPROCESS_LIST)):
            if SUBPROCESS_LIST[i][0].poll() is not None:
                old_image_path = SUBPROCESS_LIST[i][1]
                bootstrapped = SUBPROCESS_LIST[i][2]
                if bootstrapped:
                    bootstrap_pref = 'force'
                else:
                    bootstrap_pref = 'no'
                metadata = file_read_offset_metadata(
                                             old_image_path,
                                             bootstrap_pref=bootstrap_pref)
                filename = file_clean_name(old_image_path)
                results = filename + ' - ' + offset_result_str(metadata)
                main_logger.info(results)
                del SUBPROCESS_LIST[i]
                break
        if len(SUBPROCESS_LIST) == arguments.max_subprocesses:
            time.sleep(1)

    main_logger.debug('Spawning subprocess %s', str(args))
        
    pid = subprocess.Popen(args)
    SUBPROCESS_LIST.append((pid, image_path, bootstrapped))

def wait_for_all():
    while len(SUBPROCESS_LIST) > 0:
        for i in xrange(len(SUBPROCESS_LIST)):
            if SUBPROCESS_LIST[i][0].poll() is not None:
                old_image_path = SUBPROCESS_LIST[i][1]
                bootstrapped = SUBPROCESS_LIST[i][2]
                if bootstrapped:
                    bootstrap_pref = 'force'
                else:
                    bootstrap_pref = 'no'
                metadata = file_read_offset_metadata(
                                             old_image_path,
                                             bootstrap_pref=bootstrap_pref)
                filename = file_clean_name(old_image_path)
                results = filename + ' - ' + offset_result_str(metadata)
                main_logger.info(results)
                del SUBPROCESS_LIST[i]
                break
        time.sleep(1)

###############################################################################
#
# PERFORM INDIVIDUAL OFFSETS ON NAC/WAC IMAGES
#
###############################################################################

def process_offset_one_image(image_path, allow_stars=True, allow_rings=True,
                             allow_moons=True, allow_saturn=True,
                             botsim_offset=None, cartographic_data=None,
                             bootstrapped=False):
    if bootstrapped:
        bootstrap_pref = 'force'
    else:
        bootstrap_pref = 'no'
    offset_metadata = file_read_offset_metadata(image_path, overlay=False,
                                                bootstrap_pref=bootstrap_pref)
    if offset_metadata is not None:
        if not force_offset:
            if redo_offset_error:
                if 'error' not in offset_metadata:
                    main_logger.debug(
                        'Skipping %s - offset file exists and metadata OK', 
                        image_path)
                    return False
                main_logger.debug(
                    'Processing %s - offset file indicates error', image_path)
            else:
                main_logger.debug('Skipping %s - offset file exists', 
                                  image_path)
                return False
        main_logger.debug(
          'Processing %s - offset file exists but redoing offsets', image_path)
    else:
        main_logger.debug('Processing %s - no previous offset file', image_path)

    if arguments.max_subprocesses:
        run_and_maybe_wait([PYTHON_EXE, CBMAIN_OFFSET_PY] + 
                           collect_cmd_line(image_path), image_path,
                           bootstrapped) 
        return True

    if image_logfile_level != cb_logging.LOGGING_SUPERCRITICAL:
        if arguments.image_logfile is not None:
            image_log_path = arguments.image_logfile
        else:
            image_log_path = file_img_to_log_path(image_path, 'OFFSET', 
                                                  bootstrap=bootstrapped)
        
        if os.path.exists(image_log_path):
            os.remove(image_log_path) # XXX Need option to not do this
            
        image_log_filehandler = cb_logging.log_add_file_handler(
                                        image_log_path, image_logfile_level)
    else:
        image_log_filehandler = None

    image_logger = logging.getLogger('cb')
    
    image_logger.info('Command line: %s', ' '.join(command_list))
    
    try:   
        obs = file_read_iss_file(image_path)
    except:
        main_logger.exception('File reading failed - %s', image_path)
        image_logger.exception('File reading failed - %s', image_path)
        metadata = {}
        err = 'File reading failed:\n' + traceback.format_exc() 
        metadata['error'] = str(sys.exc_value)
        metadata['error_traceback'] = err
        file_write_offset_metadata(image_path, metadata)
        cb_logging.log_remove_file_handler(image_log_filehandler)
        return True

    if arguments.profile and arguments.is_subprocess:
        # Per-image profiling
        image_pr = cProfile.Profile()
        image_pr.enable()

    try:
        metadata = master_find_offset(
                              obs, create_overlay=True,
                              allow_stars=allow_stars,
                              allow_rings=allow_rings,
                              allow_moons=allow_moons,
                              allow_saturn=allow_saturn,
                              botsim_offset=botsim_offset,
                              bodies_cartographic_data=cartographic_data,
                              bootstrapped=bootstrapped,
                              stars_show_streaks=
                                 arguments.overlay_show_star_streaks,
                              stars_config=stars_config,
                              rings_config=rings_config,
                              bodies_config=bodies_config)
    except:
        main_logger.exception('Offset finding failed - %s', image_path)
        image_logger.exception('Offset finding failed - %s', image_path)
        metadata = {}
        err = 'Offset finding failed:\n' + traceback.format_exc()
        metadata['bootstrapped'] = bootstrapped 
        metadata['error'] = str(sys.exc_value)
        metadata['error_traceback'] = err
        file_write_offset_metadata(image_path, metadata)
        cb_logging.log_remove_file_handler(image_log_filehandler)
        if arguments.profile and arguments.is_subprocess:
            image_pr.disable()
            s = StringIO.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(image_pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            ps.print_callers()
            image_logger.info('Profile results:\n%s', s.getvalue())
        cb_logging.log_remove_file_handler(image_log_filehandler)
        return True
    
    try:
        file_write_offset_metadata(image_path, metadata)
    except:
        main_logger.exception('Offset file writing failed - %s', image_path)
        image_logger.exception('Offset file writing failed - %s', image_path)
        metadata = {}
        err = 'Offset file writing failed:\n' + traceback.format_exc() 
        metadata['bootstrapped'] = bootstrapped 
        metadata['error'] = str(sys.exc_value)
        metadata['error_traceback'] = err
        try:
            file_write_offset_metadata(image_path, metadata)
        except:
            main_logger.exception(
                  'Error offset file writing failed - %s', image_path)
        cb_logging.log_remove_file_handler(image_log_filehandler)
        if arguments.profile and arguments.is_subprocess:
            image_pr.disable()
            s = StringIO.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(image_pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            ps.print_callers()
            image_logger.info('Profile results:\n%s', s.getvalue())
        cb_logging.log_remove_file_handler(image_log_filehandler)
        return True

    png_image = offset_create_overlay_image(
                        obs, metadata,
                        blackpoint=arguments.png_blackpoint,
                        whitepoint=arguments.png_whitepoint,
                        gamma=arguments.png_gamma,
                        font=png_label_font)
    file_write_png_from_image(image_path, png_image,
                              bootstrap=metadata['bootstrapped'])
    
    if arguments.display_offset_results:
        display_offset_data(obs, metadata, canvas_size=None)

    if bootstrapped:
        bootstrap_pref = 'force'
    else:
        bootstrap_pref = 'no'
    metadata = file_read_offset_metadata(image_path, 
                                         bootstrap_pref=bootstrap_pref)
    filename = file_clean_name(image_path)
    results = filename + ' - ' + offset_result_str(metadata)
    main_logger.info(results)

    if arguments.profile and arguments.is_subprocess:
        image_pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(image_pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        ps.print_callers()
        image_logger.info('Profile results:\n%s', s.getvalue())

    cb_logging.log_remove_file_handler(image_log_filehandler)

    return True

#===============================================================================
# 
#===============================================================================

if arguments.profile and arguments.max_subprocesses == 0:
    # Only do image offset profiling if we're going to do the actual work in 
    # this process
    pr = cProfile.Profile()
    pr.enable()

if arguments.display_offset_results:
    root = tk.Tk()
    root.withdraw()

if arguments.use_predicted_kernels:
    iss.initialize(ck='predicted')

main_logger, image_logger = log_setup_main_logging(
               'cb_main_offset', arguments.main_logfile_level, 
               arguments.main_console_level, arguments.main_logfile,
               arguments.image_logfile_level, arguments.image_console_level)

image_logfile_level = log_decode_level(arguments.image_logfile_level)
    
force_offset = arguments.force_offset
redo_offset_error = arguments.offset_redo_error

botsim_offset = None

if arguments.botsim_offset:
    x, y = arguments.botsim_offset.split(',')
    botsim_offset = (float(x.replace('"','')), float(y.replace('"','')))

pnt_label_font = None
if arguments.png_label_font is not None:
    x, y = arguments.png_label_font.split(',')
    png_label_font = (x.replace('"',''), int(y.replace('"','')))
stars_config = STARS_DEFAULT_CONFIG.copy()
if arguments.stars_label_font is not None:
    x, y = arguments.stars_label_font.split(',')
    stars_config['font'] = (x.replace('"',''), int(y.replace('"','')))
rings_config = RINGS_DEFAULT_CONFIG.copy()
if arguments.rings_label_font is not None:
    x, y = arguments.rings_label_font.split(',')
    rings_config['font'] = (x.replace('"',''), int(y.replace('"','')))
bodies_config = BODIES_DEFAULT_CONFIG.copy()
if arguments.bodies_label_font is not None:
    x, y = arguments.bodies_label_font.split(',')
    bodies_config['font'] = (x.replace('"',''), int(y.replace('"','')))
    
if arguments.stars_only:
    arguments.allow_rings = False
    arguments.allow_moons = False
    arguments.allow_saturn = False
if arguments.rings_only:
    arguments.allow_stars = False
    arguments.allow_moons = False
    arguments.allow_saturn = False
if arguments.moons_only:
    arguments.allow_stars = False
    arguments.allow_rings = False
    arguments.allow_saturn = False
if arguments.saturn_only:
    arguments.allow_stars = False
    arguments.allow_rings = False
    arguments.allow_moons = False
        
start_time = time.time()
num_files_processed = 0
num_files_skipped = 0

main_logger.info('**********************************')
main_logger.info('*** BEGINNING MAIN OFFSET PASS ***')
main_logger.info('**********************************')
main_logger.info('')
main_logger.info('Command line: %s', ' '.join(command_list))
main_logger.info('')
main_logger.info('Subprocesses:  %d', arguments.max_subprocesses)
main_logger.info('')
main_logger.info('Allow stars:   %s', str(arguments.allow_stars))
main_logger.info('Allow rings:   %s', str(arguments.allow_rings))
main_logger.info('Allow moons:   %s', str(arguments.allow_moons))
main_logger.info('Allow Saturn:  %s', str(arguments.allow_saturn))
main_logger.info('BOTSIM offset: %s', str(botsim_offset))
main_logger.info('Pred kernels:  %s', str(arguments.use_predicted_kernels))

cartographic_data = {}

if arguments.body_cartographic_data:
    main_logger.info('')
    main_logger.info('Cartographic data provided:')
    for cart in arguments.body_cartographic_data:
        idx = cart.index('=')
        if idx == -1:
            main_logger.error('Bad format for --body-cartographic-data: %s', 
                              cart)
            sys.exit(-1)
        body_name = cart[:idx].upper()
        mosaic_path = cart[idx+1:]
        cartographic_data[body_name] = mosaic_path
        main_logger.info('    %-12s %s', body_name, mosaic_path)
    main_logger.info('')
    
    for body_name in cartographic_data:
        mosaic_path = cartographic_data[body_name]
        mosaic_metadata = file_read_mosaic_metadata_path(mosaic_path)
        if mosaic_metadata is None:
            main_logger.error('No mosaic file %s', mosaic_path)
            sys.exit(-1)
        cartographic_data[body_name] = mosaic_metadata
        
main_logger.info('')
file_log_arguments(arguments, main_logger.info)
main_logger.info('')

for image_path in file_yield_image_filenames_from_arguments(arguments):
    bootstrapped = len(cartographic_data) > 0
    if process_offset_one_image(
                    image_path,
                    allow_stars=arguments.allow_stars, 
                    allow_rings=arguments.allow_rings, 
                    allow_moons=arguments.allow_moons, 
                    allow_saturn=arguments.allow_saturn,
                    botsim_offset=botsim_offset,
                    cartographic_data=cartographic_data,
                    bootstrapped=bootstrapped):
        num_files_processed += 1
    else:
        num_files_skipped += 1

wait_for_all()

end_time = time.time()

main_logger.info('Total files processed %d', num_files_processed)
main_logger.info('Total files skipped %d', num_files_skipped)
main_logger.info('Total elapsed time %.2f sec', end_time-start_time)

if arguments.profile and arguments.max_subprocesses == 0:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    ps.print_callers()
    main_logger.info('Profile results:\n%s', s.getvalue())
