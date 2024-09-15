import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys

sys.path.append('./nn')  # implicit calling - TrajGAT
import gc
import time
import logging
import argparse
import pickle
import pynvml
import itertools
from functools import partial
from multiprocessing import Pool, Process
import numpy as np
import torch
from multiprocessing import shared_memory
from torch.utils.data.dataloader import DataLoader
from itertools import combinations
import kmedoids

from Config import Config
import tool_funcs
from utils import DatasetSynthetic, DatasetST_txt
from method_helper import *
from nn.RSTS_utils import Region  # implicit calling - RSTS
import matplotlib.pyplot as plt

'''
nohup python test_clustering_time.py --dataset xian --clustering_exp numtrajs --clustering_truth_heur_method frechet --gpu &> result &
'''


def save_clustering_results(method_name, clustering_results, output_dir):
    """
    保存聚类结果到文件中。

    参数:
    - method_name (str): 聚类方法名称。
    - num_trajs (int): 轨迹数量。
    - clustering_results (tuple): 聚类结果。
    - output_dir (str): 输出目录路径。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results_path = os.path.join(output_dir, f"{method_name}_numeral_results.pkl")

    with open(results_path, 'wb') as f:
        pickle.dump(clustering_results, f)

    logging.info(f"Clustering results saved to {results_path}")


def plot_clustering_results(method_name, clustering_results, output_dir):
    """
    绘制聚类结果并保存到文件中。

    参数:
    - method_name (str): 聚类方法名称。
    - dists (list): 轨迹距离矩阵。
    - clustering_results (tuple): 聚类结果。
    - output_dir (str): 输出目录路径。
    """
    # 提取聚类结果
    cluster_labels = clustering_results[1]  # 假设结果的第二个元素是聚类标签
    plt.figure(figsize=(10, 8))
    plt.title(f'Clustering Results - {method_name} with Trajectories')

    # 使用不同颜色显示不同聚类
    for cluster_id in set(cluster_labels):
        cluster_points = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        plt.scatter(cluster_points, [0] * len(cluster_points), label=f'Cluster {cluster_id}')

    plt.xlabel('Trajectory Index')
    plt.ylabel('Dummy Value')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, f"{method_name}_numeral_clustering.png")
    plt.savefig(plot_path)
    plt.close()

    logging.info(f"Clustering plot saved to {plot_path}")


def extract_representative_samples(dists, cluster_labels, num_representatives):
    """
    从每个聚类中提取多个代表样本。

    参数:
    - dists (list of list): 轨迹距离矩阵，每个元素表示不同轨迹之间的距离。
    - cluster_labels (array-like): 聚类标签，为每个轨迹分配一个聚类编号。
    - num_representatives (int): 每个聚类提取的代表样本数量。

    返回:
    - representative_samples (dict): 每个聚类的代表样本索引列表。
    """
    num_clusters = len(set(cluster_labels))

    # 计算每个聚类的样本
    cluster_samples = {i: [] for i in range(num_clusters)}
    for i, label in enumerate(cluster_labels):
        cluster_samples[label].append(i)

    # 计算每个聚类的代表样本
    cluster_centers = {}
    for cluster_id, indices in cluster_samples.items():
        dist_matrix = np.array([[dists[i][j] for j in indices] for i in indices])
        mean_distances = np.mean(dist_matrix, axis=1)

        # 提取前 num_representatives 个代表样本
        sorted_indices = np.argsort(mean_distances)
        cluster_centers[cluster_id] = [indices[idx] for idx in sorted_indices[:num_representatives]]

    return cluster_centers


def save_representative_samples(representative_samples, method_name, output_dir):
    """
    将代表样本保存到文件中。

    参数:
    - representative_samples (dict): 每个聚类的代表样本索引列表。
    - method_name (str): 聚类方法名称。
    - output_dir (str): 输出目录路径。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    samples_path = os.path.join(output_dir, f"{method_name}_numeral_representative_samples.pkl")
    with open(samples_path, 'wb') as f:
        pickle.dump(representative_samples, f)

    logging.info(f"Representative samples saved to {samples_path}")


def test_varying_num_trajs(datafile, seed):
    min_traj_len = Config.clustering_min_traj_len
    max_traj_len = Config.clustering_max_traj_len

    methods_s, stlabel_s = ['dtw', 'erp', 'frechet', 'hausdorff', 'T3S',
                            'TrajCL'], False
    # methods_s, stlabel_s = ['dtw', 'erp', 'frechet', 'hausdorff', 'T3S', 'TrjSR', 'TrajGAT',
    #                         'TrajCL', 'NEUTRAJ', 'MLP2'], False

    methods_st, stlabel_st = ['stedr', 'cdds', 'sar'], True

    if Config.clustering_method != '':
        if Config.clustering_method in methods_s:
            methods_s = [Config.clustering_method]
        elif Config.clustering_method in methods_st:
            methods_s = []
        else:
            methods_s = []

    output_dir = 'clustering_results'

    dataset = DatasetST_txt(datafile, seed)
    merc_range = dataset.merc_range

    for method_name in methods_s:
        shm_results = []
        base_ram, used_ram = tool_funcs.RAMInfo.mem_global(), 0
        base_gram, used_gram = tool_funcs.GPUInfo.mem()[0], 0
        logging.info("Based memory usages RAM={}, GRAM={}".format(base_ram, base_gram))

        if Config.clustering_auxilary_processor:
            shm_results = shared_memory.ShareableList([0] * 10)
            aux_processor = Process(target=hardware_usage, args=(shm_results,))
            aux_processor.start()

        batch_size = Config.effi_batch_size_gpu if Config.gpu \
            else (Config.effi_batch_size_cpu_heuristic if is_heuristic(
            method_name) else Config.effi_batch_size_cpu_learned)
        dl_num_workers = Config.effi_dataloader_num_workers
        metrics, dists = trajsimi_computation_gpucpu(method_name, dataset,
                                                     merc_range, batch_size, Config.gpu)
        print('the computation time is {}'.format(metrics))
        metrics_clus = clustering_kmedoids(dists)

        print('the clustering time is {}'.format(metrics_clus[0]))
        # 提取聚类标签
        cluster_labels = metrics_clus[1]  # 假设 metrics_clus[1] 是聚类标签
        # 提取代表样本
        representative_samples = extract_representative_samples(dists, cluster_labels, 1500)

        save_clustering_results(method_name, metrics_clus, output_dir)
        plot_clustering_results(method_name, dists, output_dir)
        # 保存代表样本
        save_representative_samples(representative_samples, method_name, output_dir)

        if Config.clustering_auxilary_processor:
            aux_processor.terminate()
            used_ram = shm_results[1] - base_ram
            used_gram = shm_results[5] - base_gram

        logging.info("[EXPFlag]exp=effi_numtrajs,dataset={},fn={},gpu={}," \
                     "time={:.4f},coltime={:.4f},embtime={:.4f},comptime={:.4f},"
                     "clustime={:.4f},ram={},gram={}".format(Config.dataset, method_name, Config.gpu,
                                                             metrics[0], metrics[1], metrics[2], metrics[3],
                                                             metrics_clus[0], used_ram, used_gram))

        if Config.clustering_auxilary_processor:
            shm_results.shm.close()
            shm_results.shm.unlink()
            del shm_results

        torch.cuda.empty_cache()
        gc.collect()

        torch.cuda.empty_cache()
        gc.collect()
    return


def trajsimi_computation_gpucpu(method_name, dataset_frame, region_range, batch_size, is_gpu):
    heuristic = is_heuristic(method_name)
    method_name = method_name + '_cpu' if (heuristic and not is_gpu) else method_name

    if heuristic:
        method_fn = heuristic_fn_wrapper(method_name)
        collate_fn = partial(heuristic_colllate_fn, is_pad=is_gpu)
    else:
        model = learned_class_wrapper(method_name, region_range)  # on target device
        method_fn = model.interpret  # on target device
        collate_fn = learned_collate_fn_wrapper(method_name, model, False)
        if not Config.gpu:
            torch.set_num_threads(min(Config.effi_batch_size_cpu_learned, Config.effi_cpu_method_num_cores))

    # dataset, _ = zip(*dataset)
    # dataset = list(dataset)

    dataset = dataset_frame.traj
    num_trajs = len(dataset)
    embs = []

    collate_time = 0.0
    embedding_time = 0.0
    computation_time = 0.0
    running_time = 0.0
    starting_ts = time.time()

    if heuristic:
        _time = time.time()
        if is_gpu:
            dataset_len = np.array(list(map(len, dataset)))
            dataset = tool_funcs.pad_lists_to_array(dataset)
        else:
            dataset_len = list(map(len, dataset))

        # idxs.shape = [2, num_trajs_q*num_trajs_d]
        # idxs = np.stack((np.array(range(num_trajs_q)).repeat(num_trajs_d), \
        # np.tile(np.array(range(num_trajs_d)), num_trajs_q)  ))
        idxs = np.array(list(combinations(range(num_trajs), 2))).T
        idxs_len = idxs.shape[1]
        collate_time += time.time() - _time

        start_idx = 0
        dists = []
        while start_idx < idxs_len:
            _time = time.time()

            end_idx = start_idx + batch_size if start_idx + batch_size < idxs_len else idxs_len
            if is_gpu:
                Q_batch = dataset[idxs[0, start_idx:end_idx]]
                D_batch = dataset[idxs[1, start_idx:end_idx]]
                Q_len_batch = dataset_len[idxs[0, start_idx:end_idx]]
                D_len_batch = dataset_len[idxs[1, start_idx:end_idx]]
            else:
                Q_batch = [dataset[_i] for _i in idxs[0, start_idx:end_idx]]
                D_batch = [dataset[_i] for _i in idxs[1, start_idx:end_idx]]
                Q_len_batch = [dataset_len[_i] for _i in idxs[0, start_idx:end_idx]]
                D_len_batch = [dataset_len[_i] for _i in idxs[1, start_idx:end_idx]]

            collate_time += time.time() - _time

            _time = time.time()
            if is_gpu:
                dists_batch = method_fn(Q_batch, Q_len_batch, D_batch, D_len_batch)
                dists_batch = dists_batch.tolist()
                dists.append(dists_batch)
            else:
                dists_batch = trajsimi_computation_heuristic_cpu(method_fn,
                                                                 Q_batch, Q_len_batch, D_batch, D_len_batch,
                                                                 num_cores=Config.effi_cpu_method_num_cores)
                dists.append(dists_batch)

            computation_time += time.time() - _time
            start_idx = end_idx

        # dists_ = sum(dists, [])
        dists_ = list(itertools.chain.from_iterable(dists))
        dists = []
        start_idx, end_idx = 0, 0
        assert len(dists_) == idxs_len
        # dists = np.ndarray([num_trajs, num_trajs]) # a
        for i_row in range(num_trajs):
            start_idx = end_idx
            end_idx += num_trajs - i_row - 1
            # dists[i_row, i_row+1 :] = dists_[start_idx : end_idx]
            dists.append([0] * (i_row + 1) + dists_[start_idx: end_idx])
        assert end_idx == idxs_len
        dists = np.array(dists)
        dists = (dists + dists.T).tolist()

    else:
        try:  # no timeout exception
            start_idx = 0
            while start_idx < num_trajs:
                _time = time.time()

                end_idx = start_idx + batch_size if start_idx + batch_size < num_trajs else num_trajs
                batch = dataset[start_idx: end_idx]
                batch_collated = collate_fn(batch)
                collate_time += time.time() - _time

                _time = time.time()
                traj_embs = method_fn(batch_collated)
                embs.append(traj_embs)
                embedding_time += time.time() - _time

                start_idx = end_idx

            embs = torch.cat(embs)
            _time = time.time()
            dists = torch.cdist(embs, embs, p=1).tolist()
            computation_time += time.time() - _time
            assert len(dists) == num_trajs

        except Exception as excpt:
            logging.error("Failed. dataset={},fn={},gpu={},error={}--{}".format(
                Config.dataset, method_name, Config.gpu,
                type(excpt), excpt))
            return [-1, -1, -1, -1], None

    if heuristic:
        # cuda.get_current_device().reset()
        pass
    else:
        del model
    running_time = time.time() - starting_ts

    return [running_time, collate_time, embedding_time, computation_time], dists


# multiprocessing-based heuristic trajsimi computation on CPU
def trajsimi_computation_heuristic_cpu(fn, lst_trajs, trajs_len, lst_trajs2, trajs_len2, num_cores):
    _time = time.time()
    n = len(lst_trajs)

    if n == 1:  # batch_size = 1: dont use multi-cores
        dists = _heuristic_cpu_operator(fn, lst_trajs, trajs_len, lst_trajs2, trajs_len2)
        assert len(dists) == n
        return dists

    slice_idx = tool_funcs.slicing(n, num_cores)
    tasks = []

    for start_idx, end_idx in slice_idx:
        tasks.append((fn, lst_trajs[start_idx: end_idx], trajs_len[start_idx: end_idx],
                      lst_trajs2[start_idx: end_idx], trajs_len2[start_idx: end_idx]))
    logging.debug("t1 = {}".format(time.time() - _time))

    pool = Pool(num_cores)
    dists = pool.starmap(_heuristic_cpu_operator, tasks)
    pool.close()

    logging.debug("t2 = {}".format(time.time() - _time))
    dists = sum(dists, [])
    assert len(dists) == n

    return dists


def _heuristic_cpu_operator(fn, lst_trajs, trajs_len, lst_trajs2, trajs_len2):
    _time = time.time()
    dists = []
    for i in range(len(lst_trajs)):
        traj1 = np.asarray(lst_trajs[i][:trajs_len[i]])
        traj2 = np.asarray(lst_trajs2[i][:trajs_len2[i]])
        dists.append(fn(traj1, traj2))
    logging.debug("t3 = {}, pid={}".format(time.time() - _time, os.getpid()))
    return dists


# def clustering_kmedoids(dists):
#     start_ts = time.time()
#
#     clustering_time = 0
#     try:
#         rtn = kmedoids.fasterpam(dists, Config.clustering_num_centroids)
#
#     except Exception as excpt:
#         logging.error("Clustering failed. dataset={},gpu={},error={}--{}".format( \
#             Config.dataset, Config.gpu, type(excpt), excpt))
#         return [-1]
#
#     clustering_time = time.time() - start_ts
#
#     return [clustering_time]
def clustering_kmedoids(dists):
    """
    Perform K-medoids clustering on the distance matrix.

    参数:
    - dists (list): 距离矩阵的列表

    返回:
    - List: [聚类时间, 聚类标签] 或 [聚类时间] 如果聚类失败
    """
    start_ts = time.time()

    clustering_time = 0
    cluster_labels = []
    try:
        # 执行 K-medoids 聚类
        rtn = kmedoids.fasterpam(dists, Config.clustering_num_centroids)
        # 假设 rtn 包含聚类标签
        cluster_labels = rtn.labels  # 确保这是正确的字段名
    except Exception as excpt:
        logging.error("Clustering failed. dataset={},gpu={},error={}--{}".format(
            Config.dataset, Config.gpu, type(excpt), excpt))
        return [-1]  # 聚类失败时的返回值

    clustering_time = time.time() - start_ts

    # 返回聚类时间和聚类标签
    return [clustering_time, cluster_labels]


# function for the async processor to record hardware usage
def hardware_usage(results):
    # results: on shared memory
    pynvml.nvmlInit()
    _h = pynvml.nvmlDeviceGetHandleByIndex(0)
    n = 0
    while True:
        time.sleep(0.01)
        n += 1
        ram = tool_funcs.RAMInfo.mem_global()
        cpu = tool_funcs.CPUInfo.usage_percent()
        gpuram = tool_funcs.GPUInfo.mem(_h)[0]
        gpupower = tool_funcs.GPUInfo.power(_h)
        results[0] = (results[0] * (n - 1) + ram) / n  # avg ram usage
        results[1] = max(results[1], ram)  # max ram usage
        results[2] = (results[2] * (n - 1) + cpu) / n  # avg cpu
        results[3] = max(results[3], cpu)  # max cpu
        results[4] = (results[4] * (n - 1) + gpuram) / n  # avg gpu ram
        results[5] = max(results[5], gpuram)  # max gpu ram
        results[6] = (results[6] * (n - 1) + gpupower) / n  # avg gpu power
        results[7] = max(results[7], gpupower)  # max gpu power


def parse_args():
    parser = argparse.ArgumentParser(description="...")
    # Font give default values here, because they will be faultly
    # overwriten by the values in config.py.
    # config.py is the correct place for default values
    parser.add_argument('--dumpfile_uniqueid', type=str, help='')  # see config.py
    parser.add_argument('--seed', type=int, help='')
    parser.add_argument('--dataset', type=str, help='')
    parser.add_argument('--gpu', dest='gpu', action='store_true', default=False)
    parser.add_argument('--clustering_exp', type=str, help='')
    parser.add_argument('--clustering_method', type=str, help='')
    parser.add_argument('--clustering_truth_heur_method', type=str, help='')

    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))


def pre_compile():  # pre-compiling the cuda-based heuristic methods, in order to avoid jit
    def _fn(methods, use_temporal):
        dataset = DatasetSynthetic(use_temporal, 100, 20, 200, 2000)
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False, drop_last=False, \
                                collate_fn=partial(heuristic_colllate_fn, is_pad=True))

        for method_name in methods:
            for batch in dataloader:
                method_fn = heuristic_fn_wrapper(method_name)
                dists = method_fn(*batch)
        return

    _fn(['dtw', 'erp', 'frechet', 'hausdorff'], False)
    _fn(['stedr', 'stlcss', 'cdds'], True)
    logging.info('pre_jit done.')


if __name__ == '__main__':
    Config.update(parse_args())

    logging.basicConfig(level=logging.DEBUG if Config.debug else logging.INFO,
                        format="[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
                        handlers=[
                            logging.FileHandler(Config.root_dir + '/exp/log/' + tool_funcs.log_file_name(), mode='w'),
                            logging.StreamHandler()])
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('dgl').setLevel(logging.ERROR)

    logging.info('python ' + ' '.join(sys.argv))
    logging.info('=================================')
    logging.info(Config.to_str())
    logging.info('=================================')

    pre_compile()

    starting_time = time.time()

    # datafile = Config.dataset_file + 'germany_20200.h5'
    datafile = '/tmp/pycharm_project_416/D_GSM_training/datasets/1-MA/train'

    test_varying_num_trajs(datafile, Config.seed)

    logging.info('all finished. @={:.1f}'.format(time.time() - starting_time))
