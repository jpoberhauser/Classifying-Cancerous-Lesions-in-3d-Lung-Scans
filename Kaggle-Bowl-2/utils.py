from socket import gethostname
import multiprocessing

def running_on_gpu_cluster():
    return gethostname() == 'ubuntu'

def get_data_dir():
    DATA_DIR = 'data/'
    # is the notebook running on the gpu-cluster
    # wish the gpu-cluster had a better name...
    if running_on_gpu_cluster():
        DATA_DIR = '/' + DATA_DIR + 'kaggle/data/'
    return DATA_DIR

def get_cpu_cores():
    cores = multiprocessing.cpu_count()
    if running_on_gpu_cluster():
        # don't get people angry
        cores /= 2

    return cores

# temp code hopefully implemented
# def get_image_paths(ids)
#     DATA_DIR = get_data_dir() + 'stage1/'
#     return [DATA_DIR + id for id in ids]
