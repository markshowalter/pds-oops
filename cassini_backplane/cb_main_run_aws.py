###############################################################################
# cb_main_run_aws.py
#
# The main top-level driver for running navigation on AWS
###############################################################################

from cb_logging import *
import logging

import argparse
import datetime
import os
import sys
import tempfile
import time
import urllib2

import boto3

import oops.inst.cassini.iss as iss
import oops

from cb_config import *
from cb_gui_offset_data import *
from cb_offset import *
from cb_util_aws import *
from cb_util_file import *


command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = '--main-console-level debug'
    
    command_list = command_line_str.split()

parser = argparse.ArgumentParser(
    description='''Cassini Backplane Main Interface for running navigation 
    on AWS''',
    epilog='''Default behavior is to feed all image names''')


###########################
### Arguments about AWS ###
###########################
parser.add_argument(
    '--retrieve-from-pds', action='store_true',
    help='''Retrieve the indexes from pds-rings.seti.org instead of 
            using the local directory structure''')
parser.add_argument(
    '--results-in-s3', action='store_true',
    help='''Store all results in the Amazon S3 cloud''')
parser.add_argument(
    '--no-update-indexes', action='store_true',
    help='''Don\'t update the index files from the PDS''')
parser.add_argument(
    '--sqs-queue-name', type=str, default='cdapsfeeder',
    help='''The name of the SQS queue; defaults to "cdapsfeeder"''')
parser.add_argument(
    '--high-water-mark', type=int, default=800,
    help='''The maximum number of entries we allow in the queue''')
parser.add_argument(
    '--low-water-mark', type=int, default=500,
    help='''The minimum number of entries we allow in the queue before 
    starting to fill again''')
parser.add_argument(
    '--drain-delay', type=int, default=10,
    help='''The number of seconds to wait between polling for queue length''')
parser.add_argument(
    '--purge-first', action='store_true',
    help='''Purge the queue before starting''')
parser.add_argument(
    '--send-done', action='store_true',
    help='''Send DONE message at end''')
parser.add_argument(
    '--done-delay', type=int, default=300,
    help='''The number of seconds to wait before sending DONE''')
parser.add_argument(
    '--aws-results-bucket', default='seti-cb-results',
    help='The Amazon S3 bucket to store results files in')
parser.add_argument(
    '--aws', action='store_true',
    help='''Set for running on AWS EC2; implies --retrieve-from-pds 
            --results-in-s3 --saturn-kernels-only --no-overlay-file
            --no-update-indexes''')

###############################
### Arguments about logging ###
###############################
parser.add_argument(
    '--main-logfile', metavar='FILENAME',
    help='''The full path of the logfile to write for the main loop; defaults 
            to $(CB_RESULTS_ROOT)/logs/cb_main_run_aws/<datetime>.log''')
LOGGING_LEVEL_CHOICES = ['debug', 'info', 'warning', 'error', 'critical', 'none']
parser.add_argument(
    '--main-logfile-level', metavar='LEVEL', default='debug', 
    choices=LOGGING_LEVEL_CHOICES,
    help='Choose the logging level to be output to the main loop logfile')
parser.add_argument(
    '--main-console-level', metavar='LEVEL', default='debug',
    choices=LOGGING_LEVEL_CHOICES,
    help='Choose the logging level to be output to stdout for the main loop')

file_add_selection_arguments(parser)

arguments = parser.parse_args(command_list)

RESULTS_DIR = CB_RESULTS_ROOT
if arguments.aws:
    arguments.retrieve_from_pds = True
    arguments.results_in_s3 = True
    arguments.no_update_indexes = True
if arguments.results_in_s3:
    RESULTS_DIR = ''


###############################################################################
#
# 
#
###############################################################################

def approximate_number_of_messages():
    attributes = AWS_SQS_CLIENT.get_queue_attributes(
                     QueueUrl=SQS_QUEUE_URL,
                     AttributeNames=['ApproximateNumberOfMessages',
                                     'ApproximateNumberOfMessagesNotVisible'])
    attributes = attributes['Attributes']
    num_msg = int(attributes['ApproximateNumberOfMessages'])
    num_msg_notvis = int(attributes['ApproximateNumberOfMessagesNotVisible'])
    main_logger.debug('In queue %d, Not vis %d, Sent %d',
                      num_msg, num_msg_notvis, num_files_processed)
    return num_msg, num_msg_notvis
    
def feed_one_image(image_path):
    global QUEUE_FEED_BEFORE_CHECKING
    if QUEUE_FEED_BEFORE_CHECKING <= 0:
        num_msg, num_msg_notvis = approximate_number_of_messages()
        if num_msg >= QUEUE_HIGH_WATER_MARK:
            main_logger.debug('Waiting for queue to drain')
            while num_msg >= QUEUE_LOW_WATER_MARK:
                time.sleep(arguments.drain_delay)
                num_msg, num_msg_notvis = approximate_number_of_messages()
            QUEUE_FEED_BEFORE_CHECKING = (QUEUE_HIGH_WATER_MARK - 
                                          QUEUE_LOW_WATER_MARK)
    QUEUE_FEED_BEFORE_CHECKING -= 1
    short_path = file_img_to_short_img_path(image_path)
    if short_path.index('_CALIB') == -1:
        short_path = short_path.replace('.IMG', '_CALIB.IMG')
    main_logger.info('=> %s', short_path)
    response = SQS_QUEUE.send_message(MessageBody=image_path)
#                                       MessageGroupId='fifo')

#===============================================================================
# 
#===============================================================================

QUEUE_NAME = arguments.sqs_queue_name

QUEUE_HIGH_WATER_MARK = arguments.high_water_mark
QUEUE_LOW_WATER_MARK = arguments.low_water_mark

QUEUE_FEED_BEFORE_CHECKING = QUEUE_HIGH_WATER_MARK

SQS_QUEUE = AWS_SQS_RESOURCE.create_queue(QueueName=QUEUE_NAME,
                                          Attributes={'MessageRetentionPeriod': 
                                                      str(14*86400),
                                                      'VisibilityTimeout': 
                                                      str(60*10), # 10 minutes
                                                     })
SQS_QUEUE_URL = SQS_QUEUE.url

main_log_path = arguments.main_logfile
main_log_path_local = main_log_path
if arguments.results_in_s3 and arguments.main_logfile_level.upper() != 'NONE':
    local_fd, main_log_path_local = tempfile.mkstemp(suffix='_mainlog.txt', 
                                                     prefix='CB_')
    os.close(local_fd)
    main_log_datetime = datetime.datetime.now().isoformat()[:-7]
    main_log_datetime = main_log_datetime.replace(':','-')
    main_log_path = 'logs/cb_main_run_aws/'+main_log_datetime
    main_log_path += '-'+str(os.getpid())+'.log'

main_logger, image_logger = log_setup_main_logging(
               'cb_main_run_aws', arguments.main_logfile_level, 
               arguments.main_console_level, main_log_path_local,
               None, None)
        
start_time = time.time()
num_files_processed = 0

main_logger.info('******************************')
main_logger.info('*** BEGINNING MAIN RUN AWS ***')
main_logger.info('******************************')
main_logger.info('')
main_logger.info('GIT Status:   %s', current_git_version())   
main_logger.info('')
main_logger.info('Command line: %s', ' '.join(command_list))
main_logger.info('')
main_logger.info('SQS Queue URL: %s', SQS_QUEUE_URL)

if arguments.retrieve_from_pds and not arguments.no_update_indexes:
    update_index_files_from_pds(main_logger)

main_logger.info('')
file_log_arguments(arguments, main_logger.info)
main_logger.info('')

if arguments.purge_first:
    main_logger.info('Purging queue')
    AWS_SQS_CLIENT.purge_queue(QueueUrl=SQS_QUEUE_URL)
    time.sleep(90) # Wait for the purge to finish
    
for image_path in file_yield_image_filenames_from_arguments(
                                                arguments,
                                                arguments.retrieve_from_pds):
    feed_one_image(image_path)
    num_files_processed += 1

if arguments.send_done:
    main_logger.info('Waiting for queue to drain before sending DONE marker')
    while True:
        num_msg, num_msg_notvis = approximate_number_of_messages()
        if num_msg > 0 or num_msg_notvis > 0:
            time.sleep(arguments.drain_delay)
            continue
        break
    main_logger.info('Delaying drain before sending DONE marker')
    time.sleep(arguments.done_delay)
    SQS_QUEUE.send_message(MessageBody='DONE')

end_time = time.time()

main_logger.info('Total files processed %d', num_files_processed)
main_logger.info('Total elapsed time %.2f sec', end_time-start_time)

if arguments.results_in_s3 and arguments.main_logfile_level.upper() != 'NONE':
    copy_file_to_s3(main_log_path_local, 
                    arguments.aws_results_bucket, main_log_path,
                    main_logger)

log_close_main_logging('cb_main_run_aws')
