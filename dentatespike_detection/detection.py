from scipy.stats import zscore
import pandas as pd
import ripple_detection as rd
import numpy as np
import pdb
import pickle
import os
import csv


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
        
        #df['trace'] = []
        #evnts_dict['traces'] = lst_trace
        df_trace = pd.DataFrame({'trace': [], 't_trace': [], 'trace_zscore': []})
        df_trace.astype(object)
        df_trace['trace'] = lst_trace
        df_trace['t_trace'] = lst_t_trace
        df_trace['trace_zscore'] = lst_zscore
        #pdb.set_trace()
        df = df.join(df_trace)
        #df = pd.concat([df, df_trace], keys=list(df.keys())+list(df_trace.keys()), axis=1, ignore_index=True)
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

#    evts = rd.core.threshold_by_zscore(data, time, min_duration, zscore_thres)
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
        
        #df['trace'] = []
        #evnts_dict['traces'] = lst_trace
        df_trace = pd.DataFrame({'trace': [], 't_trace': [], 'trace_zscore': []})
        df_trace.astype(object)
        df_trace['trace'] = lst_trace
        df_trace['t_trace'] = lst_t_trace
        df_trace['trace_zscore'] = lst_zscore
        #pdb.set_trace()
        df = df.join(df_trace)
        #df = pd.concat([df, df_trace], keys=list(df.keys())+list(df_trace.keys()), axis=1, ignore_index=True)
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
