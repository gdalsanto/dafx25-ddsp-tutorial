import torch
import torch.nn as nn 
from torch import Tensor
from flamo.auxiliary.reverb import parallelFDNAccurateGEQ, parallelFDNPEQ, parallelFirstOrderShelving
import numpy as np
import sympy as sp
import scipy
from collections import OrderedDict
from typing import List, Tuple, Literal, Union

from flamo.processor import dsp, system
from flamo.auxiliary.reverb import inverse_map_gamma, parallelFirstOrderShelving


def rt2slope(rt60: torch.Tensor, fs: int):
    r"""
    Convert time in seconds of 60 dB decay to energy decay slope.
    """
    return -60 / (rt60 * fs)


def rt2absorption(rt60: torch.Tensor, fs: int, delays_len: torch.Tensor):
    r"""
    Convert time in seconds of 60 dB decay to energy decay slope relative to the delay line length.
    """
    slope = rt2slope(rt60, fs)
    return torch.einsum("i,j->ij", slope, delays_len)

class BaseFDN(nn.Module):
    """
    FDN (Feedback Delay Network) class.

    Attributes:
        fs (int): Sampling frequency in Hz.
        nfft (int): Number of FFT points.
        N (int): Number of delay lines in the FDN.
        alias_decay_db (float): Decay rate of aliasing in decibels.
        delay_lengths (List[int]): List of delay lengths for each delay line.
        device (Literal["cpu", "cuda"]): Device to run the FDN on ("cpu" or "cuda").
    """

    def __init__(
        self,
        fs: int,
        nfft: int,
        N: int,
        alias_decay_db: float,
        delay_lengths: List[int],
        device: Literal["cpu", "cuda"],
        attenuation_type: Literal["GEQ", "lowpass"] = None,
    ) -> None:
        super().__init__()

        assert len(delay_lengths) == N

        device = torch.device(device)
        delay_lengths = torch.tensor(delay_lengths, dtype=torch.int64)

        # FDN parameters
        self.fs = fs
        self.nfft = nfft
        self.N = N
        self.alias_decay_db = alias_decay_db
        self.delay_lengths = delay_lengths
        self.device = device

        input_gain = dsp.Gain(
            size=(N, 1),
            nfft=nfft,
            requires_grad=True,
            alias_decay_db=alias_decay_db,
            device=device,
        )
        output_gain = dsp.Gain(
            size=(1, N),
            nfft=nfft,
            requires_grad=True,
            alias_decay_db=alias_decay_db,
            device=device,
        )
        delays = dsp.parallelDelay(
            size=(N,),
            max_len=delay_lengths.max(),
            nfft=nfft,
            isint=False,
            requires_grad=False,
            alias_decay_db=alias_decay_db,
            device=device,
        )
        delays.assign_value(delays.sample2s(delay_lengths))

        mixing_matrix = dsp.Matrix(
            size=(N, N),
            nfft=nfft,
            matrix_type='orthogonal',
            requires_grad=True,
            alias_decay_db=alias_decay_db,
            device=device,
        )
        if attenuation_type == None: 
            print("Generating lossless FDN")
            feedback = mixing_matrix
        else:
            attenuation = self.initialize_attenuation(attenuation_type)
            feedback = system.Series(
                OrderedDict({"mixing_matrix": mixing_matrix, "attenuation": attenuation})
            )


        # Recursion
        feedback_loop = system.Recursion(fF=delays, fB=feedback)
        fdn = system.Series(
            OrderedDict(
                {
                    "input_gains": input_gain,
                    "feedback_loop": feedback_loop,
                    "output_gains": output_gain,
                }
            )
        )

        # I/O layers
        self.input_layer = dsp.FFT(nfft)
        self.output_layer = dsp.iFFTAntiAlias(
                nfft=nfft, alias_decay_db=alias_decay_db, device=device
            )

        self.set_shell(core = fdn)

    def set_shell(self, core = None):
        self.shell = system.Shell(
            core=core,
            input_layer=self.input_layer,
            output_layer=self.output_layer,
        )

    def initialize_attenuation(self, attenuation_type):

        if attenuation_type == "GEQ":
            attenuation = dsp.parallelGEQ(
                size=(self.N,),
                octave_interval=3,
                nfft=self.nfft,
                fs=self.fs,
                requires_grad=True,
                alias_decay_db=self.alias_decay_db,
                device=self.device,
            )
            attenuation.assign_value(attenuation.param*0.4)
            attenuation.map = lambda x: 20 * torch.log10(torch.sigmoid(5*x))

        elif attenuation_type == "lowpass":
            attenuation = parallelFirstOrderShelving(
                nfft=self.nfft,
                delays=self.delay_lengths,
                device=self.device,
                fs=self.fs,
                rt_nyquist=0.5,
                alias_decay_db=self.alias_decay_db,
            )
        return attenuation
    
    def set_attenuation(self, attenuation):
        core = self.shell.get_core()
        core.feedback_loop.feedback.attenuation.assign_value(attenuation)
        self.shell.set_core(core)

    def normalize_energy(
        self,
        target_energy=1,
    ):
        """energy normalization done in the frequency domain
        Note that the energy computed from the frequency response is not the same as the energy of the impulse response
        Read more at https://pytorch.org/docs/stable/generated/torch.fft.rfft.html
        """

        H = self.shell.get_freq_response(identity=False)
        energy_H = torch.mean(torch.pow(torch.abs(H), 2))
        target_energy = torch.tensor(target_energy, device=self.device)
        # apply energy normalization on input and output gains only
        with torch.no_grad():
            core = self.shell.get_core()
            core.input_gains.assign_value(
                torch.div(
                    core.input_gains.param, torch.pow(energy_H / target_energy, 1 / 4)
                )
            )
            core.output_gains.assign_value(
                torch.div(
                    core.output_gains.param, torch.pow(energy_H / target_energy, 1 / 4)
                )
            )
            self.shell.set_core(core)

    def forward(
        self,
        inputs: Tensor,
        ext_params: List[dict],
    ) -> Tensor:
        output = []
        for x, ext_param in zip(inputs, ext_params):
            # Apply the FDN with the external parameters
            output.append(self.shell(x[..., None], ext_param))
        return torch.stack(output).squeeze(-1)
