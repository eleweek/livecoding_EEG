import mne
import numpy as np

def filter_and_drop_dead_channels(raw, picks, to_drop=None, avgref=False):
    to_drop = [] if to_drop is None else list(to_drop)

    data = raw.get_data()
    for channel in range(raw.info['nchan']):
        if np.all(data[channel] == data[channel][0]):
            to_drop.append(raw.ch_names[channel])

    raw.set_montage('standard_1020')
    raw.drop_channels(list(set(to_drop)))
        
    raw.filter(l_freq=1.0, h_freq=45.0, verbose=False)
    raw.notch_filter(50, notch_widths=4, verbose=False)

    if avgref:
        raw.set_eeg_reference('average', projection=False)

    raw.pick(picks) 