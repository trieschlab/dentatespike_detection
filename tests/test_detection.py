import numpy as np
from dentatespike_detection import detection as dect


def test_get_random_traces():
    data = np.arange(100)
    time = np.arange(100)

    n_dp = 5
    n_evts = 10
    
    df = dect.get_random_traces(
        data,
        time,
        n_dp,
        n_evts)

    assert len(df) == n_evts
    assert len(df.iloc[0]['trace_random']) == n_dp
    

def test_dist_to_neighb():
    x = np.array([1, 6, 8])
    y = np.array([3, 7])
    d_pre = np.array([-np.inf, -3, -1])
    d_fol = np.array([2, 1, np.inf])
    
    dist_pre, dist_fol = dect.dist_to_neighb(x, y)
    
    assert np.array_equal(dist_pre, d_pre)
    assert np.array_equal(dist_fol, d_fol)


def test_detect_peaks_argrelextrema():
    data = np.array([2,0,0,0,5,0,0,6])
    targ = np.array([0,0,0,0,5,0,0,0])
    
    peaks = dect.detect_peaks_argrelextrema(
        data,
        order=2,
        wndw=[-3, -1],
        rise_offset=4)
    
    ar_test = np.zeros(len(data))
    ar_test[peaks] = data[peaks]
    
    assert np.array_equal(ar_test, targ)
        