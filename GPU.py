### GPU utility functions

from subprocess import check_output

def get_num_gpus():
    """Returns the number of GPUs available
        by counting the number of devices listed by nvidia-smi.
        Note: completely not portable"""

    nvidia_smi_output = check_output("nvidia-smi")
    num_gpus = 0
    for line in nvidia_smi_output.split("\n"):
        if 'GeForce' in line:
            num_gpus += 1
    return num_gpus
