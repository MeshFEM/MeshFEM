import os.path
import sys
from subprocess import check_call
from mako.template import Template
from mako.runtime import Context

class HPCJob:
    def __init__(self, job, params):
        """
        The params should contain at least the following:
        "jobs": the job list file
        "out_dir": where to store the output
        "root_dir": where to execute the command
        "command": what command to run on each input
        "...": other parameters to set

        Note that if file/dir path is relative, it should be relative to "root_dir".
        Also note that string format will be applied (things in {} will be
        subtituted).
        """
        self.m_job = job;
        self.m_params = params;
        self.__gen_job_script();

    def run(self):
        command = "qsub {}".format(self.m_job_script);
        check_call(command.split());

    def __gen_job_script(self):
        root_dir = self.m_params["root_dir"];
        out_dir = self.m_params["out_dir"];
        basename, ext = os.path.splitext(self.m_job);
        path, name = os.path.split(basename);

        input_name = basename;
        output_name = os.path.join(out_dir, name);

        log_file = output_name + ".log"
        err_file = output_name + ".err"

        self.m_params["log_file"] = log_file
        self.m_params["err_file"] = err_file
        self.m_params["job_name"] = name
        if self.m_params["send_mail"]:
            mail_opt = "abe"
        else:
            # Only send mail on aborted jobs
            mail_opt = 'a'

        # fill in the blanks in command.
        command = self.m_params["command"].format(
                input_name = input_name,
                output_name = output_name,
                **self.m_params);

        # Generate job script
        exe_dir = sys.path[0];
        template = Template(filename=os.path.join(exe_dir, "job_template.mako"));
        job_script = os.path.join(out_dir, name + ".job");
        with open(job_script, 'w') as fout:
            ctx = Context(fout,
                    ppn = self.m_params["ppn"],
                    wall_time = self.m_params["wall_time"],
                    mem = self.m_params["mem"],
                    job_name = name,
                    log_file = log_file,
                    err_file = err_file,
                    root_dir = root_dir,
                    mail_opt = mail_opt,
                    command = command);
            template.render_context(ctx);
        self.m_job_script = job_script;
