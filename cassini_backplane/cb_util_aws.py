###############################################################################
# cb_util_aws.py
#
# Routines related to running on AWS.
#
# Exported routines:
###############################################################################

import cb_logging
import logging

import numpy as np
import socket
import subprocess

_BOTO3_AVAILABLE = True
try:
    import boto3
except ImportError:
    _BOTO3_AVAILABLE = False

from cb_util_misc import *
from cb_util_web import *

_LOGGING_NAME = 'cb.' + __name__

SQS_OFFSET_QUEUE_NAME = 'cdaps-offset-feeder'
SQS_REPROJ_BODY_QUEUE_NAME = 'cdaps-reproject-body-feeder'


AWS_PROCESSORS = {
    't2.nano': 1,
    't2.micro': 1,
    't2.small': 1,
    't2.medium': 2,
    't2.large': 2,
    't2.xlarge': 4,
    't2.2xlarge': 8,
    'm4.large': 2,
    'm4.xlarge': 4,
    'm4.2xlarge': 8,
    'm4.4xlarge': 16,
    'm4.10xlarge': 40,
    'm4.16xlarge': 64,
    'm3.medium': 1,
    'm3.large': 2,
    'm3.xlarge': 4,
    'm3.2xlarge': 8,
    'c4.large': 2,
    'c4.xlarge': 4,
    'c4.2xlarge': 8,
    'c4.4xlarge': 16,
    'c4.8xlarge': 36,
    'c3.large': 2,
    'c3.xlarge': 4,
    'c3.2xlarge': 8,
    'c3.4xlarge': 16,
    'c3.8xlarge': 32,
    'x1.16xlarge': 64,      # OK but expensive
    'x1.32xlarge': 128,     # OK but expensive
    'r4.large': 2,          # Preferred
    'r4.xlarge': 4,         # Preferred
    'r4.2xlarge': 8,        # Preferred
    'r4.4xlarge': 16,       # Preferred
    'r4.8xlarge': 32,       # Preferred
    'r4.16xlarge': 64,      # Preferred
    'r3.large': 2,          # Preferred
    'r3.xlarge': 4,         # Preferred
    'r3.2xlarge': 8,        # Preferred
    'r3.4xlarge': 16,       # Preferred
    'r3.8xlarge': 32,       # Preferred
    'i2.xlarge': 4,         # OK
    'i2.2xlarge': 8,        # OK
    'i2.4xlarge': 16,       # OK
    'i2.8xlarge': 32,       # OK
    'd2.xlarge': 4,         # OK
    'd2.2xlarge': 8,        # OK
    'd2.4xlarge': 16,       # OK
    'd2.8xlarge': 36,       # OK
    'p2.xlarge': 4,
    'p2.8xlarge': 32,
    'p2.16xlarge': 164,
    'g2.2xlarge': 8,
    'g2.8xlarge': 32
}


AWS_HOST_AMI_ID = None
AWS_HOST_PUBLIC_NAME = None
AWS_HOST_PUBLIC_IPV4 = None
AWS_HOST_INSTANCE_ID = None
AWS_HOST_INSTANCE_TYPE = None
AWS_HOST_ZONE = None
AWS_ON_EC2_INSTANCE = False
AWS_HOST_FQDN = None
AWS_S3_CLIENT= None
AWS_SQS_RESOURCE = None
AWS_SQS_CLIENT = None

if AWS_HOST_FQDN is None:
    AWS_HOST_FQDN = socket.getfqdn()
    AWS_ON_EC2_INSTANCE = AWS_HOST_FQDN.endswith('.compute.internal')
    if AWS_ON_EC2_INSTANCE:
        AWS_HOST_AMI_ID = web_read_url('http://169.254.169.254/latest/meta-data/ami-id')
        AWS_HOST_PUBLIC_NAME = web_read_url('http://169.254.169.254/latest/meta-data/public-hostname')
        AWS_HOST_PUBLIC_IPV4 = web_read_url('http://169.254.169.254/latest/meta-data/public-ipv4')
        AWS_HOST_INSTANCE_ID = web_read_url('http://169.254.169.254/latest/meta-data/instance-id')
        AWS_HOST_INSTANCE_TYPE = web_read_url('http://169.254.169.254/latest/meta-data/instance-type')
        AWS_HOST_ZONE = web_read_url('http://169.254.169.254/latest/meta-data/placement/availability-zone')

if _BOTO3_AVAILABLE:
    AWS_S3_CLIENT = boto3.client('s3')
    AWS_SQS_RESOURCE = boto3.resource('sqs')
    AWS_SQS_CLIENT = boto3.client('sqs')

def aws_copy_file_to_s3(src, bucket, dest, logger):
    logger.debug('Copying S3 %s to %s:%s', src, 
                 bucket, dest)
    AWS_S3_CLIENT.upload_file(src, bucket, dest)
    
def aws_check_for_ec2_termination():
    if AWS_ON_EC2_INSTANCE:
        term = web_read_url('http://169.254.169.254/latest/meta-data/spot/termination-time')
        if term is not None and term.find('Not Found') == -1:
            return term
    return False

def aws_add_arguments(parser, sqs_name_default, feeder=False):
    parser.add_argument(
        '--retrieve-from-pds', action='store_true',
        help='''Retrieve the image file and indexes from pds-rings.seti.org instead of 
                from the local disk''')
    parser.add_argument(
        '--no-update-indexes', action='store_true',
        help='''Don\'t update the index files from the PDS''')
    if not feeder:
        parser.add_argument(
            '--saturn-kernels-only', action='store_true',
            help='''Only load Saturn CSPICE kernels''')
    parser.add_argument(
        '--results-in-s3', action='store_true',
        help='''Store all results in the Amazon S3 cloud''')
    if not feeder:
        parser.add_argument(
            '--use-sqs', action='store_true',
            help='''Retrieve filenames from an SQS queue; ignore all file selection arguments''')
        parser.add_argument(
            '--deduce-aws-processors', action='store_true',
            help='''Deduce the number of available processors from the AMI Instance Type''')
    parser.add_argument(
        '--sqs-queue-name', type=str, default=sqs_name_default,
        help='The name of the SQS queue; defaults to "'+sqs_name_default+'"')
    parser.add_argument(
        '--aws-results-bucket', default='seti-cb-results',
        help='The Amazon S3 bucket to store results files in')
