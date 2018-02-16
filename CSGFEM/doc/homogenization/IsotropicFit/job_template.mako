#!/bin/bash
<%!
import sys
from datetime import datetime
time_stamp = datetime.now().isoformat(" ");

cmd = " ".join(sys.argv);
%>
# Script generated on ${time_stamp}
# Using command: ${cmd}

#PBS -l nodes=1:ppn=${ppn},walltime=${wall_time}
#PBS -l mem=${mem}
#PBS -N ${job_name}
#PBS -M qz263@nyu.edu
#PBS -m ${mail_opt}
#PBS -o ${log_file}
#PBS -e ${err_file}
#PBS -q s48

source /home/qz263/.bashrc
cd ${root_dir}

${command}

