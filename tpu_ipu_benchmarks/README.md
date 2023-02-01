This folder contains benchmarks for running DLRM and proposed embedding representations on TPU/IPU platforms.

First change directory in setup.sh to this directory's location and run, ". setup.sh" to configure PYTHONPATH

Then, to run a sample benchmark, try

python run_cmds/dlrm/run_dlrm.py --inference --device 'cpu'