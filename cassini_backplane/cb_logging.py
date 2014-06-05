import logging

root_logger = logging.getLogger('cb')
root_logger.setLevel(logging.DEBUG)

#fh = logging.FileHandler('spam.log')
#fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                              datefmt='%m/%d/%Y %I:%M:%S %p')
formatter = logging.Formatter('%(name)s - %(message)s')
ch.setFormatter(formatter)
#fh.setFormatter(formatter)
# add the handlers to logger
root_logger.addHandler(ch)

# Turn off FLUX loggers
flux_logger = logging.getLogger('cb.cb_util_flux')
flux_logger.propagate=False
flux_logger.setLevel(logging.ERROR)
