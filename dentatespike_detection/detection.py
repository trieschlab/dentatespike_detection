from scipy.stats import zscore
import pandas as pd
import ripple_detection as rd
import numpy as np
import pdb
import pickle
import os
import csv
from tqdm import tqdm


def detect_dentate_spikes_nokia2017(
    data,
    time,
    thres_dur,
    thres_zscore,
    thres_peak,
    thres_rise,
    wndw_rise,
    polarity=1,
    center=True,
    return_only_valid=True,
    return_trace=False,
    wdnw_trace=[-0.1, 0.1],
):
    """
    Dentate spike detection similar to Nokia et al. 2017.

    Params
    ------
    data : array_like, shape (n_time,),
    time : array_like, shape (n_time,),
    thres_zscore, float, minimal zscore,
    thres_dur : float, minimal duration of zscore crossing in s,
    thres_peak : float, minimal peak value in mV,
    thres_rise : float, minimal difference to peak in preceding window mV,
    wndw_rise : array_like, shape (2,), start and end of preceding window in s,
    polarity : signed int, 1 or -1, wether ds are positive or negative
               deflections,
    center : bool, wheter or not mean is subtracted,
    return_trace : bool, wether LFP trace is returned,
    wdnw_trace : array_like, shape (2,), start and end of trace in s,
    
    Returns
    -------
    df : pd.DataFrame, results
    """
    # adjust for polarity
    data = data*polarity
    
    if center:
        data -= np.mean(data)

#    evts = rd.core.threshold_by_zscore(data, time, min_duration, zscore_thres)
    # from github Eden-Kramer-Lab/ripple_detection/.core threshold_by_zscore
    zscored_data = zscore(data)
    is_above_threshold = zscored_data > thres_zscore
    is_above_threshold = pd.Series(is_above_threshold, index=time)
    
    evts = rd.core.segment_boolean_series(
        is_above_threshold, minimum_duration=thres_dur)

    # determine position of start and stop times in data
    tps = np.array(evts).flatten()
    pos_tps = np.searchsorted(time, tps)
    pos_tps = list(zip(pos_tps[::2],pos_tps[1::2]))

    # get peak times and amplitudes
    peaks_amp = []
    peaks_zscore = []
    peaks_t = []
    for pos_start, pos_stop in pos_tps:
        data_i = data[pos_start:pos_stop]
        peak_pos_i = np.argmax(data_i)
        peak_t_i = time[pos_start+peak_pos_i]
        peak_i = data[pos_start+peak_pos_i]
        peak_zscore_i = zscored_data[pos_start+peak_pos_i]

        # ignore events which are too close to start
        # start of preceding window must be within recording period
        if peak_t_i + wndw_rise[0] >= np.min(time):
            peaks_amp.append(peak_i)
            peaks_t.append(peak_t_i)
            peaks_zscore.append(peak_zscore_i)
    
    peaks_amp = np.array(peaks_amp)
    peaks_zscore = np.array(peaks_zscore)
    peaks_t = np.array(peaks_t)
    
    # get maximum value in time window before the spike
    ts_wndw_start = peaks_t+wndw_rise[0]
    ts_wndw_stop = peaks_t+wndw_rise[1]
    ts_wndw_both = np.sort(np.concatenate((ts_wndw_start, ts_wndw_stop)))
    pos_tps_rise = np.searchsorted(time, ts_wndw_both)
    pos_tps_rise = list(zip(pos_tps_rise[::2], pos_tps_rise[1::2]))

    peak_wndw_bfr = []
    t_peak_wndw_bfr = []
    
    for i in range(len(peaks_amp)):
        pos_start = pos_tps_rise[i][0]
        pos_stop = pos_tps_rise[i][1]
        data_i = data[pos_start:pos_stop]
        peak_pos_i = np.argmax(data_i)
        peak_i = data_i[peak_pos_i]
        peak_t_i = time[pos_start+peak_pos_i]
        
        peak_wndw_bfr.append(peak_i)
        t_peak_wndw_bfr.append(peak_t_i)

    peak_wndw_bfr = np.array(peak_wndw_bfr)
    t_peak_wndw_bfr = np.array(t_peak_wndw_bfr)
    
    # evaluate peak and rise conditions
    bool_peak = peaks_amp > thres_peak
    bool_rise = (peaks_amp-peak_wndw_bfr) > thres_rise

    bool_peak_rise = (bool_peak) & (bool_rise)
    
    peaks_amp = peaks_amp[bool_peak_rise]
    peaks_zscore = peaks_zscore[bool_peak_rise]
    peaks_t = peaks_t[bool_peak_rise]
    peak_wndw_bfr = peak_wndw_bfr[bool_peak_rise]
    t_peak_wndw_bfr = t_peak_wndw_bfr[bool_peak_rise]
    
    evnts_dict = {
        'peaks_amp': peaks_amp,
        'peaks_zscore': peaks_zscore,
        'peaks_t': peaks_t,
        'peak_wndw_bfr': peak_wndw_bfr,
        't_peak_wndw_bfr': t_peak_wndw_bfr,
    }

    df = pd.DataFrame.from_dict(evnts_dict)

    # return traces if desired
    if return_trace:
        ts_trace_start = np.array(peaks_t) + wdnw_trace[0]
        ts_trace_stop = np.array(peaks_t) + wdnw_trace[1]
        ts_trace_both = np.sort(
            np.concatenate((ts_trace_start, ts_trace_stop)))
        pos_trace = np.searchsorted(time, ts_trace_both)
        pos_trace = list(zip(pos_trace[::2], pos_trace[1::2]))
        
        lst_trace = []
        lst_t_trace = []
        lst_zscore = []
        for pos_start_i, pos_stop_i in pos_trace:
            lst_trace.append(data[pos_start_i:pos_stop_i])
            lst_t_trace.append(time[pos_start_i:pos_stop_i])
            lst_zscore.append(zscored_data[pos_start_i:pos_stop_i])
        
        df_trace = pd.DataFrame(
            {'trace': [], 't_trace': [], 'trace_zscore': []})
        df_trace.astype(object)
        df_trace['trace'] = lst_trace
        df_trace['t_trace'] = lst_t_trace
        df_trace['trace_zscore'] = lst_zscore
        df = df.join(df_trace)
    return df


def detect_dentate_spikes_zscore(
    data,
    time,
    thres_dur,
    thres_zscore_peak,
    thres_zscore_rise,
    wndw_rise,
    polarity=1,
    center=True,
    return_only_valid=True,
    return_trace=False,
    wdnw_trace=[-0.1, 0.1],
):
    
    """
    Params
    ------
    data : array_like, shape (n_time,),
    time : array_like, shape (n_time,),
    thres_zscore, float, minimal zscore,
    thres_dur : float, minimal duration of zscore crossing in s,
    thres_peak : float, minimal peak value in mV,
    thres_rise : float, minimal difference to peak in preceding window mV,
    wndw_rise : array_like, shape (2,), start and end of preceding window in s,
    polarity : signed int, 1 or -1, wether ds are positive or negative
               deflections,
    center : bool, wheter or not mean is subtracted,
    return_trace : bool, wether LFP trace is returned,
    wdnw_trace : array_like, shape (2,), start and end of trace in s,
    
    Returns
    -------
    df : pd.DataFrame, results
    """
    # adjust for polarity
    data = data*polarity
    
    if center:
        data -= np.mean(data)

    # from github Eden-Kramer-Lab/ripple_detection/.core threshold_by_zscore
    zscored_data = zscore(data)
    is_above_threshold = zscored_data > thres_zscore_peak
    is_above_threshold = pd.Series(is_above_threshold, index=time)
    
    evts = rd.core.segment_boolean_series(
        is_above_threshold, minimum_duration=thres_dur)

    # determine position of start and stop times in data
    tps = np.array(evts).flatten()
    pos_tps = np.searchsorted(time, tps)
    pos_tps = list(zip(pos_tps[::2], pos_tps[1::2]))

    # get peak times and amplitudes
    peaks_amp = []
    peaks_zscore = []
    peaks_t = []
    for pos_start, pos_stop in pos_tps:
        data_i = data[pos_start:pos_stop]
        peak_pos_i = np.argmax(data_i)
        peak_t_i = time[pos_start+peak_pos_i]
        peak_i = data[pos_start+peak_pos_i]
        peak_zscore_i = zscored_data[pos_start+peak_pos_i]

        # ignore events which are too close to start
        # start of preceding window must be within recording period
        if peak_t_i + wndw_rise[0] >= np.min(time):
            peaks_amp.append(peak_i)
            peaks_t.append(peak_t_i)
            peaks_zscore.append(peak_zscore_i)
    
    peaks_amp = np.array(peaks_amp)
    peaks_zscore = np.array(peaks_zscore)
    peaks_t = np.array(peaks_t)
    
    # get maximum value in time window before the spike
    ts_wndw_start = peaks_t+wndw_rise[0]
    ts_wndw_stop = peaks_t+wndw_rise[1]
    ts_wndw_both = np.sort(np.concatenate((ts_wndw_start, ts_wndw_stop)))
    pos_tps_rise = np.searchsorted(time, ts_wndw_both)
    pos_tps_rise = list(zip(pos_tps_rise[::2], pos_tps_rise[1::2]))

    peaks_zscore_wndw_bfr = []
    t_peak_wndw_bfr = []
    
    for i in range(len(peaks_amp)):
        pos_start = pos_tps_rise[i][0]
        pos_stop = pos_tps_rise[i][1]
        data_i = zscored_data[pos_start:pos_stop]
        peak_pos_i = np.argmax(data_i)
        peak_i = data_i[peak_pos_i]
        peak_t_i = time[pos_start+peak_pos_i]
        
        peaks_zscore_wndw_bfr.append(peak_i)
        t_peak_wndw_bfr.append(peak_t_i)

    peaks_zscore_wndw_bfr = np.array(peaks_zscore_wndw_bfr)
    t_peak_wndw_bfr = np.array(t_peak_wndw_bfr)
    
    # evaluate rise condition
    bool_rise = (peaks_zscore-peaks_zscore_wndw_bfr) > thres_zscore_rise

    peaks_amp = peaks_amp[bool_rise]
    peaks_zscore = peaks_zscore[bool_rise]
    peaks_t = peaks_t[bool_rise]
    peaks_zscore_wndw_bfr = peaks_zscore_wndw_bfr[bool_rise]
    t_peak_wndw_bfr = t_peak_wndw_bfr[bool_rise]
    
    evnts_dict = {
        'peaks_amp': peaks_amp,
        'peaks_zscore': peaks_zscore,
        'peaks_t': peaks_t,
        'peaks_zscore_wndw_bfr': peaks_zscore_wndw_bfr,
        't_peaks_zscore_wndw_bfr': t_peak_wndw_bfr,
    }

    df = pd.DataFrame.from_dict(evnts_dict)

    # return traces if desired
    if return_trace:
        ts_trace_start = np.array(peaks_t) + wdnw_trace[0]
        ts_trace_stop = np.array(peaks_t) + wdnw_trace[1]
        ts_trace_both = np.sort(
            np.concatenate((ts_trace_start, ts_trace_stop)))
        pos_trace = np.searchsorted(time, ts_trace_both)
        pos_trace = list(zip(pos_trace[::2], pos_trace[1::2]))
        
        lst_trace = []
        lst_t_trace = []
        lst_zscore = []
        for pos_start_i, pos_stop_i in pos_trace:
            lst_trace.append(data[pos_start_i:pos_stop_i])
            lst_t_trace.append(time[pos_start_i:pos_stop_i])
            lst_zscore.append(zscored_data[pos_start_i:pos_stop_i])
        
        df_trace = pd.DataFrame(
            {'trace': [], 't_trace': [], 'trace_zscore': []})
        df_trace.astype(object)
        df_trace['trace'] = lst_trace
        df_trace['t_trace'] = lst_t_trace
        df_trace['trace_zscore'] = lst_zscore
        df = df.join(df_trace)
    return df


def detect_dentate_spikes_dvorak2021(x, ):
    """
    Detect dentate spikes as in Dvorak et al., 2021, bioRxiv
    1) LFP bandpass-filter between 5-100 Hz
    2) Detect all local peaks
    3) Extract features:
        a) Amplitude difference between peak and preceding as
           well as subsequent minimum,
        b) spike width measured as distance to preceding or
           subsequent minimum, whichever is closest
        c) Z-scoring of log-transformed amplitude distribution
    4) Accept as Dentate spike if amplitudes are >.75 to preceding
       and following minima and if width of event is between 5 and 25 ms.

    Parameters
    ----------
    x : ndarray, shape (n,) | list(ndarray)
       LFP time series of length n or list of
       ndarrays if recording not continous
    
    Returns
    ----------
    """


def create_folder_structure(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' created')


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


def get_result(result):
    global results
    results.append(result)
    

def load_detect_store(
        fun_dect, fname_in, fname_out, t_start, sampling_int, p_det):
    """ Wrapper function for multiprocessing
    Load data, detect dentate spikes and store result
    """
    try:
        # open file        
        with open(fname_in, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            data = np.array(list(reader), dtype=np.float).flatten()
        # convert to mV
        data = data/1000.

        t = np.arange(t_start, t_start+len(data)*sampling_int, sampling_int)

        df_ds = fun_dect(
            data, t,
            **p_det)

        save_dict(df_ds, fname_out)
    
    except Exception as e:
        print(e)

def detect_dentate_spikes_dvorak2021(
    data,
    time,
    thres_dur,
    thres_zscore,
    thres_peak,
    thres_rise,
    wndw_rise,
    polarity=1,
    center=True,
    return_only_valid=True,
    return_trace=False,
    wdnw_trace=[-0.1, 0.1],
):
    """
    Dentate spike detection as in Dvorak et al. 2021
    
    1) Band pass filtering at 5-100 Hz
    2) Z-score signal amplitude
    3) Detection of local peak
    
    Params
    ------
    data : array_like, shape (n_time,),
    time : array_like, shape (n_time,),
    wndw_rise : array_like, shape (2,), start and end of preceding window in s,
    polarity : signed int, 1 or -1, wether ds are positive or negative
               deflections,
    center : bool, wheter or not mean is subtracted,
    return_trace : bool, wether LFP trace is returned,
    wdnw_trace : array_like, shape (2,), start and end of trace in s,
    
    Returns
    -------
    df : pd.DataFrame, results
    """


def load_dentatespikes_from_file(row, path):
    """
    Get all events stored within one file
    
    Params
    ------
    row : pandas row
    path : str, path of file
    
    Returns
    ------
    df_ds : pandas dataframe
    ------
    
    """
    animal_id = row['id']
    t_i = row['datetime']
    fname = (
        path +
        'ds_' + str(animal_id) +
        '_' + t_i.strftime("%Y_%m_%d_%H_%M_%S") + '.pkl')
    df_ds = load_dict(fname)
    return df_ds
        

def load_dentatespikes_from_df(df, path, name_trace='trace', explode=True):
    """
    Get all dentatespikes associated to rows in df
    
    Params
    ------
    df : pandas data frame
    path : str, path of files
    explode : if true, each dentatespike becomes its own row
    onlyfulllength : if true, only data with full length is permitted
    
    Returns
    ------
    df_out : pandas data frame
    ------
    
    """
    
    print('   extract all traces')
    ls_trace = []
    for i, row_i in tqdm(df.iterrows()):
        try:
            # get output name
            animal_id = row_i['id']
            t_i = row_i['datetime']
            fname = (
                path +
                'ds_' + str(animal_id) +
                '_'+t_i.strftime("%Y_%m_%d_%H_%M_%S")+'.pkl')
            df_ds = load_dict(fname)
            ls_trace.append(df_ds[name_trace])
        except:
#            pdb.set_trace()
            ls_trace.append(np.nan)
            print('Failed on: ')
            print(row_i)
    df_out = df.copy()
    df_out[name_trace] = ls_trace

    if explode:
        df_out = df_out.explode(name_trace).fillna('')
        df_out = df_out.reset_index(drop=True)
    return df_out


def ensure_entries_have_full_length(df, key, n_dp):
    """
    Discard entries which do not have a specific length

    Params
    ------
    df : pandas data frame
    key : str, name of column
    n_dp : int, number of data points per entry

    Returns
    ------
    df_out : pandas data frame
    """
    
    df_out = df.copy()
    bool_i = [len(i) == n_dp for i in df_out[key]]
    df_out = df_out[bool_i]
    df_out = df_out.reset_index(drop=True)

    return df_out


def get_random_traces(
        data,
        time,
        n_dp,
        n_evts):
    ls_trace = []
    ls_t_trace = []
    for i in range(n_evts):
        pos = np.random.choice(len(data)-n_dp)
        ls_trace.append(data[pos:pos+n_dp])
        ls_t_trace.append(time[pos:pos+n_dp])
        
    df = pd.DataFrame(
        {'trace_random': [], 't_trace': []})
    df['trace_random'] = ls_trace
    df['t_trace'] = ls_t_trace

    return df


def dist_to_neighb(x, y, check_sorted=True):
    """
    Find distance of positions in x to preceding and following
    neigbors in y. If no preceding or following neighbors
    are present, return nan.
    
    x:        1 - 6 - 8
    y:        - 3 - 7 -
    d_pre:    \ - 3 - 1
    d_fol:    2 - 1 - \
    
    
    
    Params:
    -------
    x : `ndarray`,
        positions for which distance to neighbors is to be detected
    y : `ndarray`,
        positions of neighbors
    check_sorted : bool, optional,
        verify that input is sorted and do so if it is not sorted
    
        
    Returns:
    --------
    d_pre : `ndarray`,
        distances to preceding neighbors for each position in x
    d_fol : `ndarray`,
        distances to following neighbors for each position in x
    
    """

    # check if input is sorted
    if check_sorted:
        ls = [x, y]
        for i in range(2):
            v = ls[i]
            bool_srtd = np.all(v[:-1] <= v[1:])
        if not bool_srtd:
            ls[i] = np.sort(v)
        x = ls[0]
        y = ls[1]
        
    # add -np.inf before and np.inf after to indicate those values without a valid neighbor
    y = np.concatenate([[-np.inf], y, [np.inf]])
    
    # find preceding and following neighbors
    srtd = np.searchsorted(y, x)

    val_pre = y[srtd-1]
    val_post = y[srtd]
    
    dist = [val_pre - x, val_post - x]

    return dist
