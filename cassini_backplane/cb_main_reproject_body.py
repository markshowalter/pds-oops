###############################################################################
# cb_main_reproject_body.py
#
# The main top-level driver for reprojecting a body.
###############################################################################

from cb_logging import *
import logging

import argparse
import cProfile, pstats, StringIO
import sys

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

from cb_bodies import *
from cb_config import *
from cb_gui_body_mosaic import *
import cb_logging
from cb_util_aws import *
from cb_util_file import *
from cb_util_image import *
from cb_util_web import *

MAIN_LOG_NAME = 'cb_main_reproject_body'

command_list = sys.argv[1:]

if len(command_list) == 0:
    command_line_str = '--body-name ENCELADUS N1637472791_1 --image-logfile-level debug'

    command_list = command_line_str.split()

parser = argparse.ArgumentParser(
    description='Cassini Backplane Main Interface for Reprojecting Bodies',
    epilog='''Default behavior is to reproject all bodies in all images''')

# Arguments about subprocesses
parser.add_argument(
    '--is-subprocess', action='store_true',
    help='Internal flag used to indicate this process was spawned by a parent')
parser.add_argument(
    '--max-subprocesses', type=int, default=0, metavar='NUM',
    help='The maximum number jobs to perform in parallel')

# Arguments about reprojection
parser.add_argument(
    '--offset', type=str, default='',
    help='Force an offset for the image')
parser.add_argument(
    '--use-bootstrap', action='store_true', 
    help='Use bootstrapped offset file')
parser.add_argument(
    '--force-reproject', action='store_true', 
    help='Force reprojection even if the reprojection file already exists')
parser.add_argument(
    '--display-reprojection', action='store_true', 
    help='Display the resulting reprojection')
parser.add_argument(
    '--body-name', metavar='BODY NANE',
    help='The body name to reproject in each image')
parser.add_argument(
    '--lat-resolution', metavar='N', type=float, default=0.1,
    help='The latitude resolution deg/pix')
parser.add_argument(
    '--lon-resolution', metavar='N', type=float, default=0.1,
    help='The longitude resolution deg/pix')
parser.add_argument(
    '--latlon-type', metavar='centric|graphic', default='centric',
    help='The latitude and longitude type (centric or graphic)')
parser.add_argument(
    '--lon-direction', metavar='east|west', default='east',
    help='The longitude direction (east or west)')

# Misc arguments
parser.add_argument(
    '--profile', action='store_true', 
    help='Do performance profiling')

file_add_selection_arguments(parser)
log_add_arguments(parser, MAIN_LOG_NAME, 'REPROJBODY')
aws_add_arguments(parser, SQS_REPROJ_BODY_QUEUE_NAME)
parser.add_argument(
    '--aws', action='store_true',
    help='''Set for running on AWS EC2; implies --retrieve-from-pds 
            --results-in-s3  --use-sqs --saturn-kernels-only
            --deduce-aws-processors''')

arguments = parser.parse_args(command_list)

RESULTS_DIR = CB_RESULTS_ROOT
if arguments.aws:
    arguments.retrieve_from_pds = True
    arguments.results_in_s3 = True
    arguments.saturn_kernels_only = True
    arguments.no_update_indexes = False
    arguments.use_sqs = True
    arguments.deduce_aws_processors = True
if arguments.results_in_s3:
    RESULTS_DIR = ''

if AWS_ON_EC2_INSTANCE:
    if arguments.deduce_aws_processors:
        arguments.max_subprocesses = AWS_PROCESSORS[AWS_HOST_INSTANCE_TYPE]


###############################################################################
#
# SUBPROCESS HANDLING
#
###############################################################################

def collect_cmd_line(image_path, body_name, offset):
    ret = []
    ret += ['--is-subprocess']
    ret += ['--main-logfile-level', 'none']
    ret += ['--main-console-level', 'none']
    ret += ['--image-logfile-level', arguments.image_logfile_level]
    ret += ['--image-console-level', arguments.image_console_level]
    ret += ['--force-reproject']
    if arguments.profile:
        ret += ['--profile']
    if arguments.offset is not None and arguments.offset != '':
        offset = arguments.offset
    if offset is not None:
        ret += ['--offset', '"'+offset.replace(',', ';')+'"']
    ret += ['--no-update-indexes']
    if arguments.retrieve_from_pds:
        ret += ['--retrieve-from-pds']
    if arguments.saturn_kernels_only:
        ret += ['--saturn-kernels-only']
    if arguments.results_in_s3:
        ret += ['--results-in-s3', '--aws-results-bucket', arguments.aws_results_bucket]

    ret += ['--lat-resolution', '%.3f'%arguments.lat_resolution] 
    ret += ['--lon-resolution', '%.3f'%arguments.lon_resolution] 
    ret += ['--latlon-type', arguments.latlon_type] 
    ret += ['--lon-direction', arguments.lon_direction] 

    if arguments.use_bootstrap:
        ret += ['--use-bootstrap']
    ret += ['--body-name', body_name]
    ret += ['--image-full-path', image_path]
    
    return ret

SUBPROCESS_LIST = []

def wait_for_subprocess(all=False):
    global num_files_completed
    ec2_termination_count = 0
    subprocess_count = arguments.max_subprocesses-1
    if all:
        subprocess_count = 0
    while len(SUBPROCESS_LIST) > 0:
        if ec2_termination_count == 5: # Check every 5 seconds
            ec2_termination_count = 0
            term = aws_check_for_ec2_termination()
            if term:
                # Termination notice! We have two minutes
                main_logger.error('Termination notice received - shutdown at %s',
                                  term)
                exit_processing()
        else:
            ec2_termination_count += 1
        for i in xrange(len(SUBPROCESS_LIST)):
            if SUBPROCESS_LIST[i][0].poll() is not None:
                old_image_path = SUBPROCESS_LIST[i][1]
                old_body_name = SUBPROCESS_LIST[i][2]
                old_sqs_handle = SUBPROCESS_LIST[i][3]
                filename = file_clean_name(old_image_path)
                if arguments.results_in_s3:
                    repro_path = file_clean_join(TMP_DIR,
                                                 filename+'.reproj')
                    metadata = file_read_reproj_body_path(repro_path)
                    # The repro file will already have been copied to S3
                    # in the subprocess.
                    file_safe_remove(repro_path)
                else:
                    metadata = file_read_reproj_body(
                                     old_image_path,
                                     arguments.body_name,
                                     arguments.lat_resolution*oops.RPD,
                                     arguments.lon_resolution*oops.RPD,
                                     arguments.latlon_type,
                                     arguments.lon_direction)
                num_files_completed += 1
                results = filename + ' - '
                if metadata is None:
                    results += 'REPROJ FAILED'
                else:
                    results += 'REPROJ DONE'
                results += ' (%.2f sec/repro)' % ((time.time()-start_time)/
                                                  float(num_files_completed))
                main_logger.info(results)
                del SUBPROCESS_LIST[i]
                if old_sqs_handle is not None:
                    AWS_SQS_CLIENT.delete_message(QueueUrl=SQS_QUEUE_URL,
                                                  ReceiptHandle=old_sqs_handle)
                break
        if len(SUBPROCESS_LIST) <= subprocess_count:
            break
        time.sleep(1)

def run_and_maybe_wait(args, image_path, body_name, sqs_handle):
    wait_for_subprocess()

    main_logger.debug('Spawning subprocess %s', str(args))
        
    pid = subprocess.Popen(args)
    SUBPROCESS_LIST.append((pid, image_path, body_name, sqs_handle))

#===============================================================================
# 
#===============================================================================

def reproject_image(image_path, body_name, bootstrap_pref, sqs_handle=None,
                    force_offset=None):
    repro_path = file_img_to_reproj_body_path(image_path, body_name,
                                              arguments.lat_resolution*oops.RPD,
                                              arguments.lon_resolution*oops.RPD,
                                              arguments.latlon_type,
                                              arguments.lon_direction)
    if not arguments.results_in_s3:
        if (not arguments.force_reproject and 
            os.path.exists(repro_path)):
            main_logger.info('Reproject file already exists %s', image_path)
            return False
    
    if arguments.max_subprocesses:
        run_and_maybe_wait([PYTHON_EXE, CBMAIN_REPROJECT_BODY_PY] + 
                           collect_cmd_line(image_path, body_name, 
                                            force_offset), 
                           image_path,
                           body_name, sqs_handle) 
        return True

    main_logger.info('Reprojecting %s', image_path)

    repro_metadata = None
    repro_err = None

    image_path_local = image_path
    image_path_local_cleanup = None
    image_name = file_clean_name(image_path)

    image_log_path = None
    image_log_path_local = None
    image_log_path_local_cleanup = None

    repro_path_local = None
    repro_path_local_cleanup = None

    ### Set up logging
       
    if image_logfile_level != cb_logging.LOGGING_SUPERCRITICAL:
        if arguments.image_logfile is not None:
            image_log_path = arguments.image_logfile
        else:
            image_log_path = file_img_to_log_path(
                                      image_path, 'REPROJBODY', 
                                      root=RESULTS_DIR,
                                      make_dirs=not arguments.results_in_s3)
        image_log_path_local = image_log_path
        if arguments.results_in_s3:
            image_log_path_local = file_clean_join(TMP_DIR, 
                                                   image_name+'_imglog.txt')
            image_log_path_local_cleanup = image_log_path_local
        else:
            if os.path.exists(image_log_path_local):
                os.remove(image_log_path_local) # XXX Need option to not do this

        image_log_filehandler = cb_logging.log_add_file_handler(
                                    image_log_path_local, image_logfile_level)
    else:
        image_log_filehandler = None
    # >>> image_log_path_local is live

    image_logger = logging.getLogger('cb')

    image_logger.info('Command line: %s', ' '.join(command_list))
    image_logger.info('GIT Status:   %s', current_git_version())
    image_logger.info('Reprojecting %s body %s', image_path, body_name)

    ### Download the image file if necessary

    retrieve_failed = False    
    if arguments.retrieve_from_pds:
        err, image_path_local = web_retrieve_image_from_pds(image_path,
                                                            main_logger, 
                                                            image_logger)
        image_path_local_cleanup = image_path_local 
        if err is not None:
            main_logger.error(err)
            image_logger.error(err)
            retrieve_failed = True
    # >>> image_path_local is live
    # >>> image_log_path_local is live
    
    ### Set up the repro path
    
    repro_path = file_img_to_reproj_body_path(image_path, body_name,
                                              arguments.lat_resolution*oops.RPD,
                                              arguments.lon_resolution*oops.RPD,
                                              arguments.latlon_type,
                                              arguments.lon_direction,
                                              root=RESULTS_DIR)
    repro_path_local = repro_path
    if arguments.results_in_s3:
        repro_path_local = file_clean_join(TMP_DIR, image_name+'.reproj')
        repro_path_local_cleanup = repro_path_local 
    # >>> image_path_local is live
    # >>> image_log_path_local is live
    # >>> repro_path_local is live

    ### Read the image file
    
    obs = None
    if not retrieve_failed:
        try:
            obs = file_read_iss_file(image_path_local, orig_path=image_path)
        except KeyboardInterrupt:
            raise
        except:
            main_logger.exception('File reading failed - %s', image_path)
            image_logger.exception('File reading failed - %s', image_path)
            repro_err = 'file reading failed'

    ### Then immediately delete the image file if necessary
    
    file_safe_remove(image_path_local_cleanup)
    # >>> image_log_path_local is live

    if repro_err is None:
        image_logger.info('Taken %s / %s / Size %d x %d / TEXP %.3f / %s+%s / '+
                          'SAMPLING %s / GAIN %d',
                          cspice.et2utc(obs.midtime, 'C', 0),
                          obs.detector, obs.data.shape[1], obs.data.shape[0], obs.texp,
                          obs.filter1, obs.filter2, obs.sampling,
                          obs.gain_mode)

        offset_metadata = None
        offset_str = None
        if force_offset is not None:
            offset_str = force_offset
        elif arguments.offset is not None and arguments.offset != '':
            offset_str = arguments.offset
        
        if offset_str is not None:
            offset_str_split = offset_str.split(',')
            offset = (int(offset_str_split[0]), int(offset_str_split[1]))
            offset_path = None
            image_logger.info('Forcing offset %d,%d', offset[0], offset[1])
        else:
            offset_metadata = file_read_offset_metadata(
                                                image_path, 
                                                bootstrap_pref=bootstrap_pref, 
                                                overlay=False)
            if offset_metadata is None:
                repro_err = 'no offset file',
                main_logger.error('%s - No offset file found', image_name)
                image_logger.error('%s - No offset file found', image_name)
            elif offset_metadata['status'] != 'ok':
                image_logger.info('%s - Offset file error', image_name)
                main_logger.info('%s - Offset file error', image_name) 
                repro_err = 'offset file error'
            else:
                offset = offset_metadata['offset']
                if offset is None:
                    repro_err = 'no offset'
                    image_logger.info('%s - Offset file has no valid offset', 
                                      image_name) 
                    main_logger.info('%s - Offset file has no valid offset', 
                                     image_name)
                else:
                    offset_path = offset_metadata['offset_path']
                    image_logger.info('Offset file %s', offset_path)
                    image_logger.info('Offset %d,%d', offset[0], offset[1])
#             navigation_uncertainty = metadata['model_blur_amount'] # XXX Change to Sigma
#             if navigation_uncertainty is None or navigation_uncertainty < 1.:
#                 navigation_uncertainty = 1.
#     image_logger.info('Navigation uncertainty %.2f', navigation_uncertainty)

    image_pr = None

    if repro_err is None:
        if arguments.profile and arguments.is_subprocess:
            # Per-image profiling
            image_pr = cProfile.Profile()
            image_pr.enable()

        data = image_interpolate_missing_stripes(obs.data)

        try:
            repro_metadata = bodies_reproject(
                  obs, body_name, data=data, offset=offset,
                  offset_path=offset_path,
    #               navigation_uncertainty=navigation_uncertainty, XXX
                  lat_resolution=arguments.lat_resolution*oops.RPD, 
                  lon_resolution=arguments.lon_resolution*oops.RPD,
                  latlon_type=arguments.latlon_type,
                  lon_direction=arguments.lon_direction,
                  mask_bad_areas=True)
            repro_metadata['status'] = 'ok'
        except KeyboardInterrupt:
            raise
        except:
            main_logger.exception('Reprojection failed - %s', image_path)
            image_logger.exception('Reprojection failed - %s', image_path)
            repro_err = 'reprojection failed:\n' + traceback.format_exc()
            
    if repro_err is not None:
        repro_metadata = {'status': repro_err,
                          'body_name': body_name,
                          'lat_resolution': arguments.lat_resolution*oops.RPD,
                          'lon_resolution': arguments.lat_resolution*oops.RPD,
                          'latlon_type': arguments.latlon_type,
                          'lon_direction': arguments.lon_direction}
        
    # At this point we're guaranteed to have a metadata dict
    
    repro_metadata['AWS_HOST_AMI_ID'] = AWS_HOST_AMI_ID
    repro_metadata['AWS_HOST_PUBLIC_NAME'] = AWS_HOST_PUBLIC_NAME
    repro_metadata['AWS_HOST_PUBLIC_IPV4'] = AWS_HOST_PUBLIC_IPV4
    repro_metadata['AWS_HOST_INSTANCE_ID'] = AWS_HOST_INSTANCE_ID
    repro_metadata['AWS_HOST_INSTANCE_TYPE'] = AWS_HOST_INSTANCE_TYPE
    repro_metadata['AWS_HOST_ZONE'] = AWS_HOST_ZONE

    file_write_reproj_body_path(repro_path_local, repro_metadata)

    if image_pr is not None:
        image_pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(image_pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        ps.print_callers()
        image_logger.info('Profile results:\n%s', s.getvalue())
    # >>> image_log_path_local is live
    # >>> repro_path_local is live

    cb_logging.log_remove_file_handler(image_log_filehandler)

    ### Copy results to S3
    
    if arguments.results_in_s3:
        aws_copy_file_to_s3(repro_path_local, 
                            arguments.aws_results_bucket, repro_path,
                            main_logger)
        if image_log_path_local is not None:
            aws_copy_file_to_s3(image_log_path_local, 
                                arguments.aws_results_bucket, image_log_path,
                                main_logger)

    if not arguments.is_subprocess:
        # Leave this here so it can be seen by the parent process
        file_safe_remove(repro_path_local_cleanup)
    file_safe_remove(image_log_path_local_cleanup)

    if repro_metadata['status'] == 'ok' and arguments.display_reprojection:
        mosaic_metadata = bodies_mosaic_init(body_name,
              lat_resolution=arguments.lat_resolution*oops.RPD, 
              lon_resolution=arguments.lon_resolution*oops.RPD,
              latlon_type=arguments.latlon_type,
              lon_direction=arguments.lon_direction)
        bodies_mosaic_add(mosaic_metadata, repro_metadata) 
        display_body_mosaic(mosaic_metadata, title=file_clean_name(image_path))

    return repro_metadata['status'] == 'ok'

def exit_processing():
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
    
    log_close_main_logging(MAIN_LOG_NAME)
    
    if (arguments.results_in_s3 and 
        arguments.main_logfile_level.upper() != 'NONE'):
        aws_copy_file_to_s3(main_log_path_local, 
                            arguments.aws_results_bucket, main_log_path,
                            main_logger)
        file_safe_remove(main_log_path_local)
        
    sys.exit(0)
    
#===============================================================================
# 
#===============================================================================

iss.initialize(planets=(6,))

if arguments.profile and arguments.max_subprocesses == 0:
    # Only do image offset profiling if we're going to do the actual work in 
    # this process
    pr = cProfile.Profile()
    pr.enable()

if arguments.display_reprojection:
    assert _TKINTER_AVAILABLE
    root = tk.Tk()
    root.withdraw()

iss.initialize(planets=(6,))

main_log_path = arguments.main_logfile
main_log_path_local = main_log_path
if (arguments.results_in_s3 and
    arguments.main_logfile_level.upper() != 'NONE' and
    main_log_path is None):
    main_log_path_local = '/tmp/mainlog.txt' # For CloudWatch logs
    main_log_datetime = datetime.datetime.now().isoformat()[:-7]
    main_log_datetime = main_log_datetime.replace(':','-')
    main_log_path = 'logs/'+MAIN_LOG_NAME+'/'+main_log_datetime+'-'
    if AWS_HOST_INSTANCE_ID is not None:
        main_log_path += AWS_HOST_INSTANCE_ID
    else:
        main_log_path += '-'+str(os.getpid())
    main_log_path += '.log'

main_logger, image_logger = log_setup_main_logging(
               MAIN_LOG_NAME, arguments.main_logfile_level, 
               arguments.main_console_level, main_log_path_local,
               arguments.image_logfile_level, arguments.image_console_level)

image_logfile_level = log_decode_level(arguments.image_logfile_level)
    
start_time = time.time()
num_files_processed = 0
num_files_skipped = 0
num_files_completed = 0

body_name = arguments.body_name

if arguments.use_bootstrap:
    bootstrap_pref = 'force'
else:
    bootstrap_pref = 'no'

if arguments.use_sqs:
    assert arguments.offset is None or arguments.offset == ''
    assert not arguments.display_reprojection
    assert body_name is None
         
if body_name is None and not arguments.use_sqs:
    main_logger.error('No body name specified')
    exit_processing()

if arguments.offset is not None and arguments.offset != '':
    arguments.offset = arguments.offset.replace('"','').replace(';',',')
    
main_logger.info('**********************************')
main_logger.info('*** BEGINNING REPROJECT BODIES ***')
main_logger.info('**********************************')
main_logger.info('')
main_logger.info('Host Local Name: %s', AWS_HOST_FQDN)
if AWS_ON_EC2_INSTANCE:
    main_logger.info('Host Public Name: %s (%s) in %s', AWS_HOST_PUBLIC_NAME, 
                     AWS_HOST_PUBLIC_IPV4, AWS_HOST_ZONE)
    main_logger.info('Host AMI ID: %s', AWS_HOST_AMI_ID)
    main_logger.info('Host Instance Type: %s', AWS_HOST_INSTANCE_TYPE)
    main_logger.info('Host Instance ID: %s', AWS_HOST_INSTANCE_ID)
main_logger.info('GIT Status:   %s', current_git_version())   
main_logger.info('')
main_logger.info('Command line: %s', ' '.join(command_list))
main_logger.info('')
main_logger.info('Subprocesses: %d', arguments.max_subprocesses)
main_logger.info('')
main_logger.info('Offset: %s', str(arguments.offset))

if arguments.retrieve_from_pds and not arguments.no_update_indexes:
    web_update_index_files_from_pds(main_logger)
        
if arguments.use_sqs:
    main_logger.info('')
    main_logger.info('*** Using SQS to retrieve filenames')
    main_logger.info('')
    SQS_QUEUE = None
    try:
        SQS_QUEUE = AWS_SQS_RESOURCE.get_queue_by_name(
                                           QueueName=arguments.sqs_queue_name)
    except:
        main_logger.error('Failed to retrieve SQS queue "%s"',
                          arguments.sqs_queue_name)
        exit_processing()
    SQS_QUEUE_URL = SQS_QUEUE.url
    while True:
        term = aws_check_for_ec2_termination()
        if term:
            # Termination notice! We have two minutes
            main_logger.error('Termination notice received - shutdown at %s',
                              term)
            exit_processing()
        if arguments.max_subprocesses > 0:
            wait_for_subprocess()
        messages = SQS_QUEUE.receive_messages(
                      MaxNumberOfMessages=1,
                      WaitTimeSeconds=10)
        for message in messages:
            receipt_handle = message.receipt_handle
            if message.body == 'DONE':
                # Delete it and send it again to the next instance
                AWS_SQS_CLIENT.delete_message(QueueUrl=SQS_QUEUE_URL,
                                              ReceiptHandle=receipt_handle)
                SQS_QUEUE.send_message(MessageBody='DONE')
                main_logger.info('DONE message received - exiting')
                wait_for_subprocess(all=True)
                exit_processing()
            image_path, body_name, offset = message.body.split(';')
            if reproject_image(
                            image_path,
                            body_name,
                            bootstrap_pref,
                            sqs_handle=receipt_handle,
                            force_offset=offset):
                num_files_processed += 1
                if arguments.max_subprocesses == 0:
                    AWS_SQS_CLIENT.delete_message(QueueUrl=SQS_QUEUE_URL,
                                                  ReceiptHandle=receipt_handle)
            else:
                num_files_skipped += 1
                AWS_SQS_CLIENT.delete_message(QueueUrl=SQS_QUEUE_URL,
                                              ReceiptHandle=receipt_handle)
else:
    main_logger.info('')
    file_log_arguments(arguments, main_logger.info)
    main_logger.info('')
    for image_path in file_yield_image_filenames_from_arguments(
                                                    arguments,
                                                    arguments.retrieve_from_pds):
        if reproject_image(image_path, body_name, bootstrap_pref):
            num_files_processed += 1
        else:
            num_files_skipped += 1

wait_for_subprocess(all=True)

exit_processing()

file_log_arguments(arguments, main_logger.info)
main_logger.info('')

for image_path in file_yield_image_filenames_from_arguments(arguments):
    reproject_image(image_path, body_name, bootstrap_pref)

end_time = time.time()
main_logger.info('Total elapsed time %.2f sec', end_time-start_time)

if arguments.profile:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    ps.print_callers()
    main_logger.info('Profile results:\n%s', s.getvalue())
