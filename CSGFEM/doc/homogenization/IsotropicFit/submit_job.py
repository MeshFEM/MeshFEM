#!/usr/bin/env python

import argparse
import json
import os
import os.path
import sys

from subprocess import check_call
from mako.template import Template
from mako.runtime import Context

from HPCJob import HPCJob
from LocalJob import LocalJob

def load_default_params(params):
    default_params = {
            "ppn":1,
            "wall_time": "1:00:00",
            "mem": "2GB",
            "send_mail": False};
    for key,val in default_params.iteritems():
        if key in params: continue;
        params[key] = val;

def import_jobs(job_list_file="jobs.txt"):
    """
    The job list file should contain one input names on each line.
    Line starting with # is treated as comment.
    if the path is relative, it would be relative to the directory containing
    the job list file. For example:

    # The following input file is relative to the dir containing job_list_file.
    path/a.mesh

    # The following file is absolute.
    /a/b/c.mesh
    """
    assert(os.path.exists(job_list_file));

    path = os.path.dirname(job_list_file);

    jobs = [];
    fin = open(job_list_file, 'r');
    for line in fin:
        line = line.rstrip();
        if line == "": continue;
        if line[0] == "#": continue;
        if not os.path.exists(line):
            if os.path.isabs(line):
                print("Error: %s does not exists" % line)
                continue;
            else:
                line = os.path.join(path, line)
                if not os.path.exists(line):
                    print("Error: %s does not exists" % line)
                    continue;
        jobs.append(line);
    fin.close();

    return jobs;

def identify_host():
    sysname, nodename, release, version, machine = os.uname();
    if nodename[:5] == "login":
        return "HPC";
    else:
        return "other";

def parse_argument():
    parser = argparse.ArgumentParser(description="HPC job submission script");
    parser.add_argument("job_param", help="job related parameters");
    cmd_args = parser.parse_args();
    return cmd_args;

def main():
    cmd_args = parse_argument();
    if not os.path.exists(cmd_args.job_param):
        raise IOError("{} does not exist!".format(cmd_args.job_param));

    with open(cmd_args.job_param, 'r') as fin:
        params = json.load(fin);

    load_default_params(params);

    # Taking care of relative file path
    if not os.path.isabs(params["root_dir"]):
        params["root_dir"] = os.path.abspath(
                os.path.join(
                    os.path.dirname(cmd_args.job_param),
                    params["root_dir"])
                );
    if not os.path.isabs(params["jobs"]):
        params["jobs"] = os.path.join(params["root_dir"], params["jobs"]);
    print(params["jobs"]);

    host = identify_host();
    jobs = import_jobs(params["jobs"]);
    for job_i in jobs:
        print("Submitting job: {}".format(job_i));
        if host == "HPC":
            job = HPCJob(job_i, params);
        else:
            job = LocalJob(job_i, params);
        job.run();

if __name__ == "__main__":
    main();

