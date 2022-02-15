from scipy.stats import zscore
from scipy.signal import argrelextrema
import pandas as pd
import ripple_detection as rd
import numpy as np
import pdb
import pickle
import os
import csv
from tqdm import tqdm
import warnings

def _determine_sampling_interval(time, precision):
    sampling_interval = np.median(np.diff(time))
    return sampling_interval


def detect_peaks_argrelextrema_with_info(
    data,
    time,
    width_peak_wndw=0.01,
    wndw_preceding=[-0.02, -0.01],
    rise_offset=100,
    return_trace=False,
    wdnw_trace=[-0.1, 0.1]):
    """ 
    Wrapper function around detect_peaks_argrelextrema to have
    SI units as parameters, obtain additional info and return a dataframe.
    
    Parameters
    ----------
    data : 1d, ndarray
        Array in which to find the relative extrema.
    time : ndarray
        Array with timepoints for each datapoint
    width_peak_wndw : float, optional
        width in seconds, corresponds to order parameter in detect_peaks_argrelextrema
    wndw_preceding : ndarray, optional
        timewindow in seconds, corresponds to wndw parameter in detect_peaks_argrelextrema
    rise_offset : float, optional
        determine how much peak must exceed max value in wndw_preceding
        same unit as data
    return_trace : bool, optional
        whether data trace surrounding the peak is returned
    wdnw_trace : ndarray, optional
        time window in seconds, defines size of returned trace
         
    Returns
    -------
    df : Pandas DataFrame
        With infos about each peak
    """
    
    # make sure data is 1d
    assert len(data.shape)==1
    
    # determine sampling rate
    sampling_interval = _determine_sampling_interval(time[:100], precision=10)
    
    # convert width_peak_wndw to bins
    width_peak_wndw_bins = int(np.round(width_peak_wndw/sampling_interval))
    assert width_peak_wndw_bins>0
    
    if wndw_preceding:
        # convert wndw_preceding to bins
        wndw_preceding_bins = [
            int(np.round(i/sampling_interval)) for i in wndw_preceding]
    else:
        wndw_preceding_bins = None
        
    # detect peaks
    peaks = detect_peaks_argrelextrema(
        data, width_peak_wndw_bins, wndw_preceding_bins, rise_offset)
    
    # determine peak time points and amplitudes
    peaks_t = time[peaks]
    peaks_amp = data[peaks]
    
    evnts_dict = {
        'peaks_loc': peaks,
        'peaks_amp': peaks_amp,
        'peaks_t': peaks_t,
    }

    df = pd.DataFrame.from_dict(evnts_dict)

    # return traces if desired
    if return_trace:
        
        # convert wndw_preceding to bins
        wdnw_trace_bins = [
            int(np.round(i/sampling_interval)) for i in wdnw_trace]
        
        # create empty array to store data of traces
        ar_data_trace = np.empty((len(peaks), wdnw_trace_bins[1]-wdnw_trace_bins[0]))
        ar_t_trace = np.empty((len(peaks), wdnw_trace_bins[1]-wdnw_trace_bins[0]))
        for i, shift_i in enumerate(
            range(wdnw_trace_bins[0], wdnw_trace_bins[1])):
            ar_data_trace[:, i] = data.take(
                peaks+shift_i, axis=0, mode='clip')
            ar_t_trace[:, i] = time.take(
                peaks+shift_i, axis=0, mode='clip')            
        ls_data_trace = list(ar_data_trace)
        ls_t_trace = list(ar_t_trace)
            
        df_trace = pd.DataFrame(
            {'trace': [], 't_trace': []})
        df_trace.astype(object)
        df_trace['trace'] = ls_data_trace
        df_trace['t_trace'] = ls_t_trace
        df = df.join(df_trace)
    
    return df

    
def detect_peaks_argrelextrema(
    data,
    order=10,
    wndw=[-10, -5],
    rise_offset=100
    ):
    """
    detect_peaks uses a two step process to find sharp transitions
    in one dimensional time series data. 
     1) Candidate peaks are identified using the argrelextrema of 
        scipy.signal with the order parameter.
        This functions detects relative extrema in 1D data which
        exceed their n neighbours on each sides. n is determined by the
        order parameter. 
     2) The value of each candidate peak must exceed the maximum plus a 
        rise_offset for any value in a given timewindow before its occurence.
        This ought to ensure sharp onset. 
        
    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    order : int, optional
        order of scipy.signal.argrelextrema
    wndw : ndarray, optional
         list or array of length 2. Defines window in which each value is
         compared to peak
    rise_offset : float, optional
         Offset added to values in window.
         
    Returns
    -------
    locs : ndarray
        Location of peaks
    
    """
    # define params for take functions in both steps
    axis = 0
    mode = 'clip'
    
    # find candidate extrema
    locs = argrelextrema(
        data, np.greater, order=order, axis=axis, mode=mode)[0]
    
    if wndw and rise_offset:
        # keep only if peak exceeds maxima in preceding window.
        results = np.ones(len(locs), dtype=bool)
        main = data.take(locs, axis=axis, mode=mode)
        for shift in range(wndw[0], wndw[1]):
            minus = data.take(locs + shift, axis=axis, mode=mode)
            minus = minus + rise_offset
            results &= np.greater(main, minus)
            if(~results.any()):
                break
        locs = locs[np.nonzero(results)]
    return locs


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
        print('Warning: searchsorted on concatenated start and stops can'+
              'create problems with overlapping windows.')
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
    
    # create empty dataframe for all cases in which no
    # spikes are detected
    evnts_dict = {
        'peaks_amp': [],
        'peaks_zscore': [],
        'peaks_t': [],
        'peaks_zscore_wndw_bfr': [],
        't_peaks_zscore_wndw_bfr': [],
    }
    df = pd.DataFrame.from_dict(evnts_dict)
    
    # interrupt if there are no events detected
    if len(evts)==0:
        return df

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
        try:
            peak_pos_i = np.argmax(data_i)
        except:
            pdb.set_trace()
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
    pos_wndw_start = np.searchsorted(time, ts_wndw_start)
    pos_wndw_stop = np.searchsorted(time, ts_wndw_stop)
    pos_tps_rise = list(zip(pos_wndw_start, pos_wndw_stop))

    peaks_zscore_wndw_bfr = []
    t_peak_wndw_bfr = []
    
    for i in range(len(peaks_amp)):
        try:
            pos_start = pos_tps_rise[i][0]
            pos_stop = pos_tps_rise[i][1]
            data_i = zscored_data[pos_start:pos_stop]
            peak_pos_i = np.argmax(data_i)
            peak_i = data_i[peak_pos_i]
            peak_t_i = time[pos_start+peak_pos_i]
        except:
            pdb.set_trace()
        
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
