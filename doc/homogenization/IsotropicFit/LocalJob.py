import os.path
import sys
from subprocess import call

class LocalJob:
    def __init__(self, job, params):
        self.m_job = job;
        self.m_params = params;

    def run(self):
        root_dir = self.m_params["root_dir"];
        out_dir = self.m_params["out_dir"];
        basename, ext = os.path.splitext(self.m_job);
        path, name = os.path.split(basename);

        input_name = basename;
        output_name = os.path.join(out_dir, name);

        command = self.m_params["command"].format(
                input_name = input_name,
                output_name = output_name,
                **self.m_params);

        call(command.split());

