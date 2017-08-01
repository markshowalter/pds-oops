Running navigation on AWS EC2 Instances
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Copying results from S3 to the local disk
=========================================

cd <aws directory>
rm -rf *
aws s3 sync s3://seti-cb-results .
rsync -avh --progress . $CB_RESULTS_ROOT

