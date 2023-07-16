# Kalman Filter Implementation

A Straight-forward Kalman Filter Implementation in both `numpy` and `pytorch`.

This is a reference implementation of Kalman Filter for performance optimisation purpose.

- [PyTorch Implementaation](./kalman_filter.py)
- [Numpy Implementation](./kalman_filter_np.py)

## Kalman Filter Usage

We can simply compare the result with the following script.

```python
from kalman_filter_np import KalmanFilter as NumpyKalman
from kalman_filter import KalmanFilter as TorchKalman
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# The Ground Truth Reference of pykalman
def KalmanFilterBaseline(x):
    from pykalman import KalmanFilter
  # Construct a Kalman filter
    kf = KalmanFilter(transition_matrices = [1],
    observation_matrices = [1],
    initial_state_mean = 0,
    initial_state_covariance = 1,
    observation_covariance=1,
    transition_covariance=.01)
  # Use the observed values of the price to get a rolling mean
    state_means, _ = kf.filter(x)
    return state_means

kf = NumpyKalman(n_feature=1)
kf2 = TorchKalman(n_feature=1)
x = np.array([i for i in range(1, 1001)])

# Create the Ground Truth Data
ground_data = np.log(x)
# Create the Source Data
src_data = ground_data + np.random.randn(1000)

# Create Kalman Filter Mean
smooth_data = kf.filter(src_data.reshape(-1, 1)[:500])
smooth_data_c = kf2.filter(src_data.reshape(-1, 1)[:500])
smooth_data2 = KalmanFilterBaseline(src_data)

# Forward Update Test
for i in range(500, len(src_data), 1):
    new_data, _ = kf.step(src_data[i])
    smooth_data = np.append(smooth_data, new_data)
    smooth_data_c = np.append(smooth_data_c, kf2.step(src_data[i]))

plt.figure(figsize=(10, 5), dpi=200)
plt.plot(src_data, label='src')
plt.plot(smooth_data, label='smooth')
plt.plot(smooth_data2, label='smooth2')
plt.plot(smooth_data_c, label='smoothc')
plt.plot(ground_data, label='ground', c='r')
plt.legend()
plt.show()
```