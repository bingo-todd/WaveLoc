import numpy as np
from BasicTools import get_fpath


def file_reader(reverb_set_dir, batch_size=128, is_shuffle=True,
                frame_len=320, shift_len=160, n_azi=37):
    """ read wav files in given directies, one file per time
    Args:
        record_set_dir: directory or list of directories where recordings exist
        batch_size:
        is_shuffle:
    Returns:
        samples generator, [samples, label_all]
    """
    if isinstance(reverb_set_dir, list):
        dir_all = reverb_set_dir
    else:
        dir_all = [reverb_set_dir]
    #
    fpath_reverb_all = []
    for dir_fpath in dir_all:
        fpath_all_tmp = get_fpath(dir_fpath, '.npy', is_absolute=True)
        fpath_reverb_all.extend(fpath_all_tmp)

    if is_shuffle:
        np.random.shuffle(fpath_reverb_all)

    for fpath_reverb in fpath_reverb_all:
        x_d_batch, x_r_batch, y_loc_batch, is_anechoic = np.load(fpath_reverb,allow_pickle=True)
        # if x_d.shape[0] == batch_size and x_r.shape[0] == batch_size and y_loc.shape[0] == batch_size:
        yield x_r_batch, y_loc_batch
    