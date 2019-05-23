"""
Script en python que hace llamada como tipo de script bash, para llamar otro script python en multiples nodos.
"""
from subprocess import run as sp__run, PIPE as sp__PIPE, Popen as sp__Popen
from datetime import datetime as dt__datetime
from shlex import split as sl__split
from re import search as re__search
import time

import logging
logger = logging.getLogger('stdout_only')


def send_work(Instance, group_info, base_BDs_names, gral_params, w_time):
    """
    Passes almost all arguments to manage function via sbatch script.
    w_time (waitting time) per node bases.
    """
    nth_group = group_info[0]
    job_name = Instance.simulation_name
    working_dir = Instance.Working_dir

    # sbatch output formats for logging
    stdout_name_f = "slurm_{}_{}.stdout"
    stderr_name_f = "slurm_{}_{}.stderr"
    # set arguments (coma separated)
    Args = ["tuple({})".format(group_info)]  # always tuple
    Args += ["list({})".format(base_BDs_names)]  # always list
    Args += ["list({})".format(gral_params)]  # always list

    # create sbatch script to send to nodes
    sbatch_cmd = "sbatch -J {} -D {} -o {} -e {} --wrap "
    sbatch_cmd = sbatch_cmd.format(
        job_name,
        working_dir,
        stdout_name_f.format(job_name, '%j'),
        stderr_name_f.format(job_name, '%j'),
    )
    python_cmd = 'module load python; python -c "'
    python_cmd += 'from smcfpl.core_calc import in_node_manager;'
    python_cmd += 'in_node_manager({args})"'
    python_cmd = python_cmd.format(args=','.join(Args))

    # msg = "RUN Command:\n{}".format(sbatch_cmd + python_cmd)
    # lunch full sbatch script to node
    sbatch_cmd = sl__split(sbatch_cmd) + [python_cmd]
    output_cmd = sp__run(sbatch_cmd, shell=False, stdout=sp__PIPE, stderr=sp__PIPE)
    # logger.debug(msg)

    # fetch job number
    job_id = output_cmd.stdout.decode('utf-8')
    searched = re__search(r'Submitted batch job ([0-9]+)', job_id)
    if searched:
        # if not None, file was send successfully
        job_id = searched.group(1)
    else:
        # find in the error output. print what happened
        out_err_msg = output_cmd.stderr.decode('utf-8')
        logger.error(out_err_msg)
        raise Exception(out_err_msg)

    # reconstruct corresponding outputs file names
    stdout_fname = stdout_name_f.format(job_name, job_id)
    stderr_fname = stderr_name_f.format(job_name, job_id)

    # get node name of job id
    time.sleep(0.7)  # it's to fast. Wait for job allocation
    nodenom_cmd = """squeue --jobs={} --format="%N" --noheader""".format(job_id)
    node_name = sp__run(sl__split(nodenom_cmd), shell=False, stdout=sp__PIPE).stdout.decode().rstrip('\n')
    msg = "Waiting response for group {} (from Node: {}) ...".format(nth_group, node_name)
    logger.info(msg)

    # waits for output file to appear (up to waiting_time; w_time). Checks directory every some seconds for the files.
    T_now = dt__datetime.now()
    while dt__datetime.now() - T_now < w_time:
        # waits for a seconds so slurm can allocate jobs properly and re read jobs
        time.sleep(1)
        # verifies if job was allocated, find job in queue. Remember it will not be allocated if insuficient nodes are available
        bash_cmd1 = sl__split("squeue --noheader --jobs={}".format(job_id))
        bash_cmd2 = sl__split("wc -l")
        cmd1 = sp__Popen(bash_cmd1, shell=False, stdout=sp__PIPE)
        cmd2 = sp__run(bash_cmd2, shell=False, stdin=cmd1.stdout, stdout=sp__PIPE)
        break_cond = int(cmd2.stdout)
        if not break_cond:
            # supposely when not anymore, it's finished.
            msg = "group {}/{} (job: {}) finished!".format(group_info[0], group_info[2], job_id)
            logger.info(msg)
            break
    else:  # executed when while condition becomes false (e.g., after loop)
        msg = "Timeout reached!. Cancelling job: {}".format(job_id)
        # read and print everything outputed up to now? (looks like scancel overwrites log file)
        logger.warn(msg)
        sp__run(["scancel", "{}".format(job_id)])

    # return status values(?) - (n_cases_succeded, n_stages_succeded)
    return job_id
