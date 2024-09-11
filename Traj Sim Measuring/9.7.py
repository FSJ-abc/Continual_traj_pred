import os
import time
import numpy as np
from tqdm import tqdm
import csv


def read_file(file_path, delim='\t'):
    """Read data from a file and return as a NumPy array."""
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(file_path, 'r') as file:
        header = file.readline()  # Skip header
        data = [list(map(float, line.strip().split(delim))) for line in file]
    return np.asarray(data)


def save_memory_as_csv(distances, file_path):
    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write distances to the CSV file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp_ms', 'track_id', 'x', 'y'])  # Write header
        num = 0
        timestamp_ms = 100
        for track in distances:
            for point in track[1:]:  # Skip the first element which is trackId
                writer.writerow([timestamp_ms, num, point[0], point[1]])  # Write each point
                timestamp_ms = timestamp_ms + 100
            num = num + 1
            timestamp_ms = 100


def compute_distance(p, q):
    """Compute DTW distance using a recursive approach."""
    len_p, len_q = len(p), len(q)
    ca = np.full((len_p, len_q), -1.0)

    def _c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        if i == 0 and j == 0:
            ca[i, j] = np.linalg.norm(p[i] - q[j])
        elif i > 0 and j == 0:
            ca[i, j] = max(_c(i - 1, 0), np.linalg.norm(p[i] - q[j]))
        elif i == 0 and j > 0:
            ca[i, j] = max(_c(0, j - 1), np.linalg.norm(p[i] - q[j]))
        elif i > 0 and j > 0:
            ca[i, j] = max(
                min(
                    _c(i - 1, j),
                    _c(i - 1, j - 1),
                    _c(i, j - 1)
                ),
                np.linalg.norm(p[i] - q[j])
            )
        else:
            ca[i, j] = float('inf')
        return ca[i, j]

    return _c(len_p - 1, len_q - 1)


def main():
    data_dir = '../D-GSM-training/datasets/3-ZS/train'
    all_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
    case_num = 0
    memory = []
    seq_idx = [0, 0]
    print("Processing Data .....")
    start_time = time.time()
    pbar = tqdm(total=len(all_files))
    for path in all_files:
        track_list = []
        data = read_file(path, '\t')
        tracks = np.unique(data[:, 1])
        for track in tracks:
            track_list.append(data[data[:, 1] == track, 2:])
        seq_num = len(track_list)
        max_distance = [0, 0]

        if case_num == 0:
            for i in range(len(track_list)):
                p = track_list[i]
                for j in range(i, len(track_list)):
                    q = track_list[j]
                    if len(p) == 0 or len(q) == 0:
                        raise ValueError('Input sequences are empty.')

                    # Adjust coordinates
                    p[:, 0] += (q[0, 0] - p[0, 0])
                    p[:, 1] += (q[0, 1] - p[0, 1])

                    dist = compute_distance(p, q)
                    if max_distance[0] > dist:
                        max_distance[0] = dist
                        seq_idx[1] = j
                        seq_idx[0] = i
            memory.append(track_list[seq_idx[0]])
            memory.append(track_list[seq_idx[1]])
        elif len(memory) < 300:
            for i in range(len(memory)):
                p = memory[i]
                for j in range(i, len(track_list)):
                    q = track_list[j]
                    if len(p) == 0 or len(q) == 0:
                        raise ValueError('Input sequences are empty.')

                        # Adjust coordinates
                    p[:, 0] += (q[0, 0] - p[0, 0])
                    p[:, 1] += (q[0, 1] - p[0, 1])
                    dist = compute_distance(p, q)
                    if max_distance[0] > dist:
                        max_distance[0] = dist
                        seq_idx[1] = j
                memory.append(track_list[seq_idx[1]])

        else:
            for i in range(len(track_list)):
                p = track_list[i]
                for j in range(i, len(memory)):
                    q = memory[j]
                    if len(p) == 0 or len(q) == 0:
                        raise ValueError('Input sequences are empty.')

                        # Adjust coordinates
                    p[:, 0] += (q[0, 0] - p[0, 0])
                    p[:, 1] += (q[0, 1] - p[0, 1])
                    dist = compute_distance(p, q)
                    if max_distance[0] > dist:
                        max_distance[0] = dist
                        seq_idx[1] = i
            memory.append(track_list[seq_idx[1]])

        pbar.update(1)
        case_num = case_num + 1
    pbar.close()
    save_memory_as_csv(memory, './memory/memory3-ZS')

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    # print(distances)


if __name__ == '__main__':
    main()
