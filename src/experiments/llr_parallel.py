#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Autoreload packages that are modified
# get_ipython().magic(u'load_ext autoreload')
# get_ipython().magic(u'autoreload 2')

# Load relevant packages
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn import *
import sys
import subprocess
from datetime import datetime, timedelta
import netCDF4
import time
from functools import partial
import os

if os.path.basename(os.getcwd()) == "experiments":
    os.chdir(os.path.join("..",".."))

# Adds 'experiments' folder to path to load experiments_util
sys.path.insert(0, 'src/experiments')
# Load general utility functions
from experiments_util import *
# Load functionality for fitting and predicting
from fit_and_predict import *
# Load functionality for evaluation
from skill import *
# Load functionality for stepwise regression
from stepwise_util import *
from backward_stepwise_pll import backward_stepwise

# In[3]:


#
# Choose experiment parameters
#
gt_id = "contest_tmp2m" # "contest_precip" or "contest_tmp2m"
target_horizon = "34w" # "34w" or "56w"
margin_in_days = 56
criterion = "mean"

# If run_locally is False, forecast generation jobs for each target date 
# are submitted to a batch cluster using batch_script (recommended)
# If run_locally is True, forecast generation for each target data is 
# executed locally and sequentially and the setting of batch_script is irrelevant
run_locally = True
# Shell script for submitting batch job to cluster; please change to your personal 
# batch cluster submission script.
# Usage for our script is:
#   src/batch/quick_sbatch_python script.py\ script_arg1\ script_arg2 num_cores mem
batch_script = 'src/batch/quick_sbatch_python.sh'
num_cores = 16
mem = "20GB"

contest_id = get_contest_id(gt_id, target_horizon)

#
# Create list of submission dates in YYYYMMDD format
#
submission_dates = [datetime(y,4,18)+timedelta(14*i) for y in range(2011,2018) for i in range(26)]
submission_dates = ['{}{:02d}{:02d}'.format(date.year, date.month, date.day) for date in submission_dates]

procedure = "backward_stepwise"
hindcast_features = False
num_cores = 8

submission_dates_new = []
for submission_date_str in submission_dates:
    # Load result file name for checking convergence for this submission date
    file_name = default_result_file_names(
        gt_id = gt_id, 
        target_horizon = target_horizon, 
        margin_in_days = margin_in_days,
        criterion = criterion,
        submission_date_str = submission_date_str,
        procedure = "backward_stepwise",
        hindcast_folder = False,
        hindcast_features = False,
        use_knn1 = False)["converged"]
    file_name = file_name.replace("contest_period", "2011-2018")
#     print file_name
    
    if not os.path.exists(file_name):
    	import pdb
        submission_dates_new.append(submission_date_str)

subset = submission_dates_new[0:num_cores]

n = len(subset)
gt_ids = n*[gt_id]
target_horizons = n*[target_horizon]
margin_in_days_itr = n*[margin_in_days]
criterions = n*[criterion]
hindcast_features_itr = n*[hindcast_features]
num_cores_itr = n*[num_cores]


Parallel(n_jobs=-1)(delayed(backward_stepwise)(gt_id, target_horizon, margin_in_days, criterion, hindcast_features, submission_date_str, num_cores) for 
gt_id, target_horizon, margin_in_days, criterion, hindcast_features, submission_date_str, num_cores in zip(gt_ids, target_horizons, margin_in_days_itr, criterions, hindcast_features_itr, subset, num_cores_itr))







