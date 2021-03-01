import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def plot_series(time, series):
    plt.figure(figsize=(10, 6))
    plt.plot(time, series)
    plt.xlabel('time')
    plt.ylabel('value')
    plt.grid(True)
    plt.show()


def trend(time, slope=0.0):
    return slope * time


# 生成季节性的时间序列
def seasonal_pattern(season_time):
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


# 加入噪声
def noise(time, noise_level=1):
    return np.random.randn(len(time)) * noise_level


# auto-correlation
def autocorrelation(time, amplitude):  # amplitude:振幅
    rho1 = 0.8
    ar = np.random.randn(len(time) + 1)
    for step in range(1, len(time) + 1):
        ar[step] += rho1 * ar[step - 1]
    return ar[1:] * amplitude


if __name__ == '__main__':
    # 周期性信号 + 随机信号 + 趋势信号

    time = np.arange(4 * 365 + 1)  # [0  1  2 ... 1458 1459 1460]

    # series = trend(time, 0.1)
    # plot_series(time, series)

    period = 365
    amplitude = 40
    # series = seasonality(time, period, amplitude)
    # plot_series(time, series)

    baseline = 10
    slope = 0.05
    series = baseline + trend(time, slope) + seasonality(time, period, amplitude)
    # plot_series(time, series)

    noise_level = 40  # 或 40
    noisy_series = series + noise(time, noise_level)
    # plot_series(time, noisy_series)

    series = autocorrelation(time, amplitude=10)
    # plot_series(time[:200], series[:200])

    series = autocorrelation(time, amplitude=10) + trend(time, slope=2)
    # plot_series(time[:200], series[:200])

    series = autocorrelation(time, amplitude=10) + trend(time, slope=2) + \
             seasonality(time, 50, amplitude=150)
    plot_series(time[:200], series[:200])
