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
    
