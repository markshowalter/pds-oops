###############################################################################
# cb_main_offset.py
#
# The main top-level driver for single file offset finding.
###############################################################################

from cb_logging import *
import logging

import argparse
import cProfile, pstats, StringIO
import datetime
import os
import subprocess
import sys
import tempfile
import time
import traceback
import urllib2

_BOTO3_AVAILABLE = True
try:
    import boto3
except ImportError:
    _BOTO3_AVAILABLE = False

_TKINTER_AVAILABLE = True
try:
    import Tkinter as tk
except ImportError:
    _TKINTER_AVAILABLE = False

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

#     command_line_str = 'W1569203537_1 --force-offset --image-console-level debug --display-offset-results' # Single bright star - Vega
#     command_line_str = 'N1470361604_1 --force-offset --image-console-level debug --display-offset-results' # Single bright star - Vega CLR
#     command_line_str = 'N1470361637_1 --force-offset --image-console-level debug --display-offset-results' # Single bright star - Vega GRN
#     command_line_str = 'N1470361670_1 --force-offset --image-console-level debug --display-offset-results' # Single bright star - Vega UV3
#     command_line_str = 'N1470361703_1 --force-offset --image-console-level debug --display-offset-results' # Single bright star - Vega BL2
#     command_line_str = 'N1470361736_1 --force-offset --image-console-level debug --display-offset-results' # Single bright star - Vega MT2

#     command_line_str = 'N1470497510_1 --force-offset --image-console-level debug --display-offset-results' # Thinks Saturn fills the frame? XXX
#     command_line_str = 'N1485868397_1 --force-offset --image-console-level debug --display-offset-results' # Single bright star - Fomalhaut UV3
    command_line_str = 'N1492103550_3 --force-offset --image-console-level debug --display-offset-results' # Single bright star - EpsOri GRN
#     command_line_str = 'N1492103583_3 --force-offset --image-console-level debug --display-offset-results' # Single bright star - EpsOri RED
#     command_line_str = 'N1503229507_1 --force-offset --image-console-level debug --display-offset-results' # Single bright star - AlpSco CLR
#     command_line_str = 'N1508714071_1 --force-offset --image-console-level debug --display-offset-results' # Single bright star - AlpLyr UV3
#     command_line_str = 'N1514820484_1 --force-offset --image-console-level debug --display-offset-results' # Single bright star - TheTau CB3
#     command_line_str = 'N1514839259_1 --force-offset --image-console-level debug --display-offset-results' # Single bright star - TheTau IR4
#     command_line_str = 'N1521513816_1 --force-offset --image-console-level debug --display-offset-results' # Single bright star - BetOri GRN
#     command_line_str = 'N1521514002_1 --force-offset --image-console-level debug --display-offset-results' # Single bright star - BetOri MT3
#     command_line_str = 'N1521514663_1 --force-offset --image-console-level debug --display-offset-results' # Single bright star - BetOri CB3
#     command_line_str = 'N1548433221_1 --force-offset --image-console-level debug --display-offset-results' # Single bright star - TheTau RED,IR1
#     command_line_str = 'N1548521082_1 --force-offset --image-console-level debug --display-offset-results' # Single bright star - KapCet GRN
#     command_line_str = 'N1569375917_1 --force-offset --image-console-level debug --display-offset-results' # Single bright star - TheTau IR1
#     command_line_str = 'N1578052095_1 --force-offset --image-console-level debug --display-offset-results' # Single bright star - AlpSco CLR

#     command_line_str = 'N1595336241_1 --force-offset --image-console-level debug --display-offset-results' # Star smear with edge of A ring
#    command_line_str = 'N1751425716_1 --force-offset --image-console-level debug --display-offset-results' # Star smear with nothing else
#     command_line_str = 'N1484580522_1 --force-offset --image-console-level debug --display-offset-results --stars-label-font courbd.ttf,30 --png-label-font courbd.ttf,30' # Star smear with Mimas

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
#    command_line_str = 'N1669812089_1 --max-subprocesses 1 --retrieve-from-pds --results-in-s3 --main-console-level debug --image-console-level debug --force-offset'
#     command_line_str = 'W1507076645_1 --image-console-level debug --main-console-level debug --force-offset --retrieve-from-pds'#--no-wac-offset'
#    command_line_str = '--volume COISS_2099 --main-console-level info --image-console-level none --image-logfile-level none --aws --max-subprocesses 2'
    
    command_list = command_line_str.split()

parser = argparse.ArgumentParser(
    description='Cassini Backplane Main Interface for Offsets',
    epilog='''Default behavior is to perform an offset pass on all images
              without associated offset files''')

###XXXX####
# --image-logfile is incompatible with --max-subprocesses > 0


###########################
### Arguments about AWS ###
###########################
parser.add_argument(
    '--retrieve-from-pds', action='store_true',
    help='''Retrieve the image file from pds-rings.seti.org instead of 
            from the local disk''')
parser.add_argument(
    '--saturn-kernels-only', action='store_true',
    help='''Only load Saturn CSPICE kernels''')
parser.add_argument(
    '--results-in-s3', action='store_true',
    help='''Store all results in the Amazon S3 cloud''')
parser.add_argument(
    '--no-update-indexes', action='store_true',
    help='''Don\'t update the index files from the PDS''')
parser.add_argument(
    '--aws', action='store_true',
    help='''Set for running on AWS EC2; implies --retrieve-from-pds 
            --results-in-s3 --saturn-kernels-only --no-overlay-file
            --no-update-indexes''')
parser.add_argument(
    '--aws-results-bucket', default='seti-cb-results',
    help='The Amazon S3 bucket to store results files in')

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
    '--redo-error', action='store_true', default=False,
    help='''Force offset computation if the offset file exists and 
            indicates a fatal error''')
parser.add_argument(
    '--redo-non-spice-error', action='store_true', default=False,
    help='''Force offset computation if the offset file exists and 
            indicates a fatal error other than missing SPICE data''')
parser.add_argument(
    '--redo-spice-error', action='store_true', default=False,
    help='''Force offset computation if the offset file exists and 
            indicates a fatal error from missing SPICE data''')
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
parser.add_argument(
    '--no-wac-offset', action='store_true', default=False,
    help='Don\'t use the computed offset between NAC and WAC frames')

#######################################################
### Arguments about overlay and PNG file generation ###
#######################################################
parser.add_argument(
    '--overlay-show-star-streaks', action='store_true', default=False,
    help='Show star streaks in the overlay and PNG files')
parser.add_argument(
    '--no-overlay-file', action='store_true', default=False,
    help='Don\'t generate an overlay file')
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

RESULTS_DIR = CB_RESULTS_ROOT
if arguments.aws:
    arguments.retrieve_from_pds = True
    arguments.results_in_s3 = True
    arguments.saturn_kernels_only = True
    arguments.no_overlay_file = True
    arguments.no_update_indexes = True
if arguments.results_in_s3:
    RESULTS_DIR = ''


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
    ret += ['--no-update-indexes']
    if arguments.retrieve_from_pds:
        ret += ['--retrieve-from-pds']
    if arguments.saturn_kernels_only:
        ret += ['--saturn-kernels-only']
    if arguments.results_in_s3:
        ret += ['--results-in-s3', '--aws-results-bucket', arguments.aws_results_bucket]
    if arguments.no_overlay_file:
        ret += ['--no-overlay-file']
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
    if arguments.png_blackpoint:
        ret += ['--png-blackpoint', str(arguments.png_blackpoint)]
    if arguments.png_whitepoint:
        ret += ['--png-whitepoint', str(arguments.png_whitepoint)]
    if arguments.png_gamma:
        ret += ['--png-gamma', str(arguments.png_gamma)]
    if arguments.png_label_font:
        ret += ['--png-label-font', arguments.png_label_font]
    if arguments.stars_label_font:
        ret += ['--stars-label-font', arguments.stars_label_font]
    if arguments.rings_label_font:
        ret += ['--rings-label-font', arguments.rings_label_font]
    if arguments.bodies_label_font:
        ret += ['--bodies-label-font', arguments.bodies_label_font]
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
                filename = file_clean_name(old_image_path)
                if arguments.results_in_s3:
                    results = filename + ' - Job completed'
                else:
                    if bootstrapped:
                        bootstrap_pref = 'force'
                    else:
                        bootstrap_pref = 'no'
                    metadata = file_read_offset_metadata(
                                                 old_image_path,
                                                 bootstrap_pref=bootstrap_pref)
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
                filename = file_clean_name(old_image_path)
                if arguments.results_in_s3:
                    results = filename + ' - Job completed'
                else:
                    if bootstrapped:
                        bootstrap_pref = 'force'
                    else:
                        bootstrap_pref = 'no'
                    metadata = file_read_offset_metadata(
                                                 old_image_path,
                                                 bootstrap_pref=bootstrap_pref)
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

def copy_file_to_s3(src, dest):
    main_logger.debug('Copying S3 %s to %s:%s', src, 
                      arguments.aws_results_bucket, dest)
    s3 = boto3.client('s3')
    s3.upload_file(src, arguments.aws_results_bucket, dest)
    
def process_offset_one_image(image_path, allow_stars=True, allow_rings=True,
                             allow_moons=True, allow_saturn=True,
                             botsim_offset=None, cartographic_data=None,
                             bootstrapped=False):
    if bootstrapped:
        bootstrap_pref = 'force'
    else:
        bootstrap_pref = 'no'
        
    if not arguments.results_in_s3:
        file_exists = os.path.exists(file_img_to_offset_path(image_path))
        if file_exists:
            if not force_offset:
                if (redo_offset_error or redo_offset_nonspice_error or
                    redo_offset_spice_error):
                    offset_metadata = file_read_offset_metadata(
                                                image_path, overlay=False,
                                                bootstrap_pref=bootstrap_pref)
                    if 'error' not in offset_metadata:
                        main_logger.debug(
                            'Skipping %s - offset file exists and metadata OK', 
                            image_path)
                        return False
                    if redo_offset_error: 
                        main_logger.debug(
                            'Processing %s - offset file indicates error', image_path)
                    else:
                        assert redo_offset_nonspice_error or redo_offset_spice_error
                        error = offset_metadata['error']
                        if error == '':
                            error = offset_metadata['error_traceback'].split('\n')[-2]
                        if error.startswith('SPICE(NOFRAMECONNECT)'):
                            if redo_offset_spice_error:
                                main_logger.debug(
                                    'Processing %s - offset file indicates SPICE error', image_path)
                            else:
                                main_logger.debug(
                                    'Skipping %s - offset file indicates SPICE error', image_path)
                                return False
                        else:
                            if redo_offset_spice_error:
                                main_logger.debug(
                                    'Skipping %s - offset file indicates non-SPICE error', image_path)
                                return False
                            else:
                                main_logger.debug(
                                    'Processing %s - offset file indicates non-SPICE error', image_path)
                else:
                    main_logger.debug('Skipping %s - offset file exists', 
                                      image_path)
                    return False
            else:
                main_logger.debug(
                  'Processing %s - offset file exists but redoing offsets', image_path)
        else:
            main_logger.debug('Processing %s - no previous offset file', image_path)
    else:
        main_logger.debug('Processing %s - Using S3 storage', image_path)

    if arguments.max_subprocesses:
        run_and_maybe_wait([PYTHON_EXE, CBMAIN_OFFSET_PY] + 
                           collect_cmd_line(image_path), image_path,
                           bootstrapped) 
        return True

    image_path_local = image_path
    
    if arguments.retrieve_from_pds:
        short_path = file_img_to_short_img_path(image_path)
        if short_path.index('_CALIB') == -1:
            short_path = short_path.replace('.IMG', '_CALIB.IMG')
        url = PDS_RINGS_CALIB_ROOT + short_path
        try:
            local_fd, image_path_local = tempfile.mkstemp(suffix='.IMG', prefix='ISS_')
            main_logger.debug('Retrieving %s to %s', url, image_path_local)
            url_fp = urllib2.urlopen(url)
            os.write(local_fd, url_fp.read())
            url_fp.close()
            os.close(local_fd)
        except urllib2.HTTPError, e:
            main_logger.error('Failed to retrieve %s: %s', url, e)
            try:
                os.remove(image_path_local)
            except:
                pass
            return False
        except urllib2.URLError, e:
            main_logger.error('Failed to retrieve %s: %s', url, e)
            try:
                os.remove(image_path_local)
            except:
                pass
            return False
        
    image_log_path_local = None
    if image_logfile_level != cb_logging.LOGGING_SUPERCRITICAL:
        if arguments.image_logfile is not None:
            image_log_path = arguments.image_logfile
        else:
            image_log_path = file_img_to_log_path(
                                      image_path, 'OFFSET', 
                                      bootstrap=bootstrapped,
                                      root=RESULTS_DIR,
                                      make_dirs=not arguments.results_in_s3)
        image_log_path_local = image_log_path
        if arguments.results_in_s3:
            local_fd, image_log_path_local = tempfile.mkstemp(
                                                  suffix='_imglog.txt', prefix='CB_')
            os.close(local_fd)
        else:
            if os.path.exists(image_log_path_local):
                os.remove(image_log_path_local) # XXX Need option to not do this

        image_log_filehandler = cb_logging.log_add_file_handler(
                                        image_log_path_local, image_logfile_level)
    else:
        image_log_filehandler = None

    image_logger = logging.getLogger('cb')
    
    image_logger.info('Command line: %s', ' '.join(command_list))

    try:
        obs = file_read_iss_file(image_path_local, orig_path=image_path)
    except:
        main_logger.exception('File reading failed - %s', image_path)
        image_logger.exception('File reading failed - %s', image_path)
        metadata = {}
        err = 'File reading failed:\n' + traceback.format_exc()
        metadata['status'] = 'error'
        metadata['status_detail1'] = str(sys.exc_value)
        metadata['status_detail2'] = err
        file_write_offset_metadata(image_path, metadata)
        cb_logging.log_remove_file_handler(image_log_filehandler)
        if arguments.retrieve_from_pds:
            try:
                os.remove(image_path_local)
            except:
                pass
        if arguments.results_in_s3:
            copy_file_to_s3(offset_path_local, offset_path)
            if image_log_path_local is not None:
                copy_file_to_s3(image_log_path_local, image_log_path)
                try:
                    os.remove(image_log_path_local)
                except:
                    pass
            try:
                os.remove(offset_path_local)
            except:
                pass
        return True
    
    if arguments.retrieve_from_pds:
        try:
            os.remove(image_path_local)
        except:
            pass

    if obs.dict['SHUTTER_STATE_ID'] == 'DISABLED':
        main_logger.info('Skipping because shutter disabled - %s', image_path)
        image_logger.info('Skipping because shutter disabled - %s', 
                          image_path)
        metadata = {}
        metadata['status'] = 'skipped'
        metadata['status_detail1'] = 'Shutter disabled'
        metadata['status_detail2'] = 'Shutter disabled'
        file_write_offset_metadata(image_path, metadata)
        cb_logging.log_remove_file_handler(image_log_filehandler)
        if arguments.results_in_s3:
            copy_file_to_s3(offset_path_local, offset_path)
            copy_file_to_s3(image_log_path_local, image_log_path)
            try:
                os.remove(offset_path_local)
            except:
                pass
            try:
                os.remove(image_log_path_local)
            except:
                pass
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
        metadata['status'] = 'error' 
        metadata['status_detail1'] = str(sys.exc_value)
        metadata['status_detail2'] = err
        file_write_offset_metadata(image_path, metadata)
        if arguments.profile and arguments.is_subprocess:
            image_pr.disable()
            s = StringIO.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(image_pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            ps.print_callers()
            image_logger.info('Profile results:\n%s', s.getvalue())
        cb_logging.log_remove_file_handler(image_log_filehandler)
        if arguments.results_in_s3:
            copy_file_to_s3(offset_path_local, offset_path)
            if image_log_path_local is not None:
                copy_file_to_s3(image_log_path_local, image_log_path)
                try:
                    os.remove(image_log_path_local)
                except:
                    pass
            try:
                os.remove(offset_path_local)
            except:
                pass
        return True

    offset_path = file_img_to_offset_path(
                              image_path, 
                              bootstrap=bootstrapped,
                              root=RESULTS_DIR,
                              make_dirs=not arguments.results_in_s3)
    offset_path_local = offset_path
    if arguments.results_in_s3:
        local_fd, offset_path_local = tempfile.mkstemp(suffix='.OFF', 
                                                       prefix='CB_')
        os.close(local_fd)
    overlay_path_local = None
    if not arguments.no_overlay_file:
        overlay_path = file_img_to_overlay_path(
                                    image_path, 
                                    bootstrap=bootstrapped,
                                    root=RESULTS_DIR,
                                    make_dirs=not arguments.results_in_s3)
        overlay_path_local = overlay_path
        if arguments.results_in_s3:
            local_fd, overlay_path_local = tempfile.mkstemp(suffix='.OVR', prefix='CB_')
            os.close(local_fd)
    
    try:
        file_write_offset_metadata_path(offset_path_local, metadata,
                                        overlay=not arguments.no_overlay_file,
                                        overlay_path=overlay_path_local)
    except:
        main_logger.exception('Offset file writing failed - %s', image_path)
        image_logger.exception('Offset file writing failed - %s', image_path)
        metadata = {}
        err = 'Offset file writing failed:\n' + traceback.format_exc() 
        metadata['bootstrapped'] = bootstrapped
        metadata['status'] = 'error'
        metadata['status_detail1'] = str(sys.exc_value)
        metadata['status_detail2'] = err
        try:
            # It seems weird to try again, but it might work with less metadata
            file_write_offset_metadata_path(offset_path_local, metadata,
                                            overlay=False)
        except:
            main_logger.exception(
                  'Error offset file writing failed - %s', image_path)
        if arguments.profile and arguments.is_subprocess:
            image_pr.disable()
            s = StringIO.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(image_pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            ps.print_callers()
            image_logger.info('Profile results:\n%s', s.getvalue())
        cb_logging.log_remove_file_handler(image_log_filehandler)
        if arguments.results_in_s3:
            copy_file_to_s3(offset_path_local, offset_path)
            if image_log_path_local is not None:
                copy_file_to_s3(image_log_path_local, image_log_path)
                try:
                    os.remove(image_log_path_local)
                except:
                    pass
            try:
                os.remove(offset_path_local)
            except:
                pass
            if not arguments.no_overlay_file:
                try:
                    os.remove(overlay_path_local)
                except:
                    pass
        return True

    png_path = file_img_to_png_path(
                            image_path, 
                            bootstrap=bootstrapped,
                            root=RESULTS_DIR,
                            make_dirs=not arguments.results_in_s3)
    png_path_local = png_path
    if arguments.results_in_s3:
        local_fd, png_path_local = tempfile.mkstemp(suffix='.PNG', prefix='CB_')
        os.close(local_fd)

    png_image = offset_create_overlay_image(
                        obs, metadata,
                        blackpoint=arguments.png_blackpoint,
                        whitepoint=arguments.png_whitepoint,
                        gamma=arguments.png_gamma,
                        font=png_label_font)
    file_write_png_path(png_path_local, png_image)
    
    if arguments.display_offset_results:
        display_offset_data(obs, metadata, canvas_size=None)

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

    if arguments.results_in_s3:
        copy_file_to_s3(offset_path_local, offset_path)
        if image_log_path_local is not None:
            copy_file_to_s3(image_log_path_local, image_log_path)
            try:
                os.remove(image_log_path_local)
            except:
                pass
        copy_file_to_s3(png_path_local, png_path)
        try:
            os.remove(offset_path_local)
        except:
            pass
        if not arguments.no_overlay_file:
            try:
                os.remove(overlay_path_local)
            except:
                pass
        try:
            os.remove(png_path_local)
        except:
            pass

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
    assert _TKINTER_AVAILABLE
    root = tk.Tk()
    root.withdraw()

if arguments.results_in_s3:
    assert _BOTO3_AVAILABLE
    
kernel_type = 'reconstructed'
if arguments.use_predicted_kernels:
    kernel_type = 'predicted'
iss.initialize(ck=kernel_type, planets=(6,),
               offset_wac=not arguments.no_wac_offset)

main_log_path = arguments.main_logfile
main_log_path_local = main_log_path
if arguments.results_in_s3 and arguments.main_logfile_level.upper() != 'NONE':
    local_fd, main_log_path_local = tempfile.mkstemp(suffix='_mainlog.txt', 
                                                     prefix='CB_')
    os.close(local_fd)
    main_log_datetime = datetime.datetime.now().isoformat()[:-7]
    main_log_datetime = main_log_datetime.replace(':','-')
    main_log_path = 'logs/cb_main_offset/'+main_log_datetime
    main_log_path += '-'+str(os.getpid())+'.log'

main_logger, image_logger = log_setup_main_logging(
               'cb_main_offset', arguments.main_logfile_level, 
               arguments.main_console_level, main_log_path_local,
               arguments.image_logfile_level, arguments.image_console_level)

image_logfile_level = log_decode_level(arguments.image_logfile_level)
    
force_offset = arguments.force_offset
redo_offset_error = arguments.redo_error
redo_offset_nonspice_error = arguments.redo_non_spice_error
redo_offset_spice_error = arguments.redo_spice_error

botsim_offset = None

if arguments.botsim_offset:
    x, y = arguments.botsim_offset.split(',')
    botsim_offset = (float(x.replace('"','')), float(y.replace('"','')))

png_label_font = None
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

if arguments.retrieve_from_pds and not arguments.no_update_indexes:
    main_logger.info('Downloading PDS index files')
    index_no = 2001
    while True:
        index_file = os.path.join(COISS_2XXX_DERIVED_ROOT,
                                  'COISS_%04d-index.tab' % index_no)
        if not os.path.exists(index_file):
            url = PDS_RINGS_VOLUMES_ROOT + (
                            'COISS_%04d/index/index.tab' % index_no)
            main_logger.debug('Trying %s', url)
            try:
                url_fp = urllib2.urlopen(url)
                index_fp = open(index_file, 'w')
                index_fp.write(url_fp.read())
                index_fp.close()
                url_fp.close()
            except urllib2.HTTPError, e:
                main_logger.info('Failed to retrieve %s: %s', url, e)
                break
            except urllib2.URLError, e:
                main_logger.info('Failed to retrieve %s: %s', url, e)
                break
        index_no += 1
        
main_logger.info('')
file_log_arguments(arguments, main_logger.info)
main_logger.info('')

for image_path in file_yield_image_filenames_from_arguments(
                                                arguments,
                                                arguments.retrieve_from_pds):
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

log_close_main_logging('cb_main_offset')

if arguments.results_in_s3 and arguments.main_logfile_level.upper() != 'NONE':
    copy_file_to_s3(main_log_path_local, main_log_path)
