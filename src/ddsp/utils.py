
import matplotlib.pyplot as plt
from numpy.typing import NDArray

import numpy as np
import soundfile as sf
import sympy as sp

def plot_time_domain(x: NDArray, fs: int, xlog: bool = False, title:str ="Time Domain Signal", xlabel:str ="Time (s)", ylabel:str ="Amplitude"):
    """
    Plot the time-domain signal.

    Parameters
    ----------
    x : array_like
        The signal to plot.
    fs : int or float
        The sampling frequency of the signal in Hz.
    title : str, optional
        The title of the plot (default is "Time Domain Signal").
    xlabel : str, optional
        The label for the x-axis (default is "Time (s)").
    ylabel : str, optional
        The label for the y-axis (default is "Amplitude").

    Returns
    -------
    None
        This function displays the plot and does not return any value.
    """
    time = np.arange(len(x)) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(time, x)
    if xlog:
        plt.xscale('log')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0, time[-1])
    plt.grid()
    plt.show()


def plot_spectrogram(x: NDArray, fs: int, title="Spectrogram", n_fft=1024, hop_length=None, clim = [-100, 0], cmap="viridis"):
    """
    Plot the spectrogram of a signal.

    Parameters
    ----------
    x : array_like
        The input signal.
    fs : int or float
        The sampling frequency of the signal in Hz.
    title : str, optional
        The title of the plot (default is "Spectrogram").
    n_fft : int, optional
        Number of FFT points (default: 1024).
    hop_length : int, optional
        Number of samples between successive frames (default: n_fft // 4).
    clim : list of float, optional
        Color limits for the spectrogram in dB (default: [-100, 0]).
    cmap : str, optional
        Colormap for the spectrogram (default: "viridis").

    Returns
    -------
    None
        This function displays the plot and does not return any value.
    """
    if hop_length is None:
        hop_length = n_fft // 4
    plt.figure(figsize=(10, 4))
    S, freqs, bins, im = plt.specgram(x, NFFT=n_fft, Fs=fs, noverlap=n_fft - hop_length, cmap=cmap)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(im).set_label('Intensity [dB]')
    plt.clim(clim[0], clim[1])
    plt.tight_layout()
    plt.show()

def audioread(rir_path: str, to_mono: bool = True) -> tuple[NDArray, int]:
    """
    Read an audio file and optionally convert it to mono.

    Parameters
    ----------
    rir_path : str
        Path to the audio file.
    to_mono : bool, optional
        If True, convert stereo to mono by averaging channels (default: True).

    Returns
    -------
    tuple[np.ndarray, int]
        Tuple containing the audio data and sampling rate.
    """
    rir, fs = sf.read(rir_path)
    if rir.ndim > 1 and to_mono:
        rir = rir.mean(axis=1)
    return rir, fs


def find_onset(rir: NDArray) -> int:
    """
    Find the onset in a room impulse response (RIR) by extracting a local energy envelope and locating its maximum.

    Parameters
    ----------
    rir : np.ndarray
        Room impulse response of shape num_time_samples x num_channels

    Returns
    -------
    int
        Index of the detected onsets in the RIRs.
    """
    win_len = 64
    overlap = 0.75
    win = np.hanning(win_len)[:, np.newaxis]

    if len(rir.shape) == 1:
        rir = np.expand_dims(rir, -1)
    # pad rir
    pad_width = int(win_len * overlap)
    rir = np.pad(rir, ((pad_width, pad_width), (0, 0)))
    hop = 1 - overlap
    hop_len = int(win_len * hop)
    n_wins = int(np.floor(rir.shape[0] / hop_len - 1 / (2 * hop)))

    local_energy = []
    for i in range(1, n_wins - 1):
        start = (i - 1) * hop_len
        end = start + win_len
        segment = rir[start:end, :]
        if segment.shape[0] != win_len:
            continue
        energy_per_channel = np.sum((segment**2) * win,
                                    axis=0)  # shape num_channels,
        local_energy.append(energy_per_channel)

    # convert to 2D array of shape (num_windows, num_channels)
    local_energy = np.stack(local_energy, axis=0)

    # discard trailing points
    n_win_discard = int((overlap / hop) - (1 / (2 * hop)))
    local_energy = local_energy[n_win_discard:, :]
    if len(local_energy) == 0:
        return 0
    onset_idx = np.argmax(local_energy, axis=0)
    return (win_len * hop * (onset_idx - 1)).astype(int)

def find_coprime_numbers(min_value, max_value, num_numbers, target_sum):
    def gcd(a, b):
        while b != 0:
            a, b = b, a % b
        return a

    def is_coprime(a, b):
        return gcd(a, b) == 1

    def get_logarithmically_distributed_values(min_value, max_value, num_numbers):
        # Generate logarithmically distributed values between min_value and max_value
        log_min = np.log(min_value)
        log_max = np.log(max_value)
        log_values = np.linspace(log_min, log_max, num_numbers)
        values = np.exp(log_values).astype(int)
        return np.unique(values)  # Remove duplicates and return

    def get_closest_numbers(values, reference_values):
        closest_numbers = []
        for i in values:
            closest_numbers.append(min(reference_values, key=lambda x: abs(x - i)))
        if len(np.unique(closest_numbers)) != len(closest_numbers):
            print("Warning: sampling duplicate values")
            
        return closest_numbers

    # Start with logarithmically distributed values
    prime_numbers = list(sp.primerange(min_value, max_value))
    log_values = get_logarithmically_distributed_values(min_value, max_value, num_numbers)
    coprime_numbers = get_closest_numbers(log_values, prime_numbers)

    current_sum = sum(coprime_numbers)
    while current_sum < 0.9 * target_sum or current_sum > target_sum * 1.1:  # allow 2% error
        if current_sum < target_sum:
            coprime_numbers = [sp.nextprime(num)  for num in coprime_numbers]
        else:
            coprime_numbers = [sp.prevprime(num)  for num in coprime_numbers]
        current_sum = sum(coprime_numbers)
        
    return coprime_numbers  