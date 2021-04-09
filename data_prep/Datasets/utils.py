import os


def get_bad_indices(dir_path, pairs_list,):
    bad_indices = []

    for i, pair in enumerate(pairs_list):
        for item in pair:
            path = os.path.join(dir_path, item)
            if not os.path.isfile(path):
                bad_indices.append(i)
    bad_indices.sort(reverse=True)
    return bad_indices
