### GPU utility functions

def get_num_gpus():
    """Returns the number of GPUs available"""

    from pycuda import driver 
    driver.init()
    num_gpus = driver.Device.count()
    return num_gpus
