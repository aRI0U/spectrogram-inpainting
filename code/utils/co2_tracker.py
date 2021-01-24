import time

import torch
if torch.cuda.is_available():
    import nvidia_smi


class CO2Tracker:
    def __init__(self, gpus=[]):
        r"""CO2 consumption tracker for deep learning models.
        Look at https://arxiv.org/abs/1906.02243 for details.
        """
        # temporal variables
        self._start = None
        self._step = None

        # power variables
        self._cpu_power = 0
        self._gpu_power = 0
        self._ram_power = 0
        self.total_energy = 0

        # GPU-specific constants
        self._cuda = torch.cuda.is_available()
        print(gpus)
        if self._cuda:
            nvidia_smi.nvmlInit()
            self._handles = [nvidia_smi.nvmlDeviceGetHandleByIndex(gpu) for gpu in gpus]

        # energy consumption constants
        self._pue_coeff = 1.58
        self._co2_coeff = 0.477

    @property
    def co2_equivalent(self):
        return self._co2_coeff * self.total_energy

    def power_stats(self):
        return dict(
            cpu=self._cpu_power,
            gpu=self._gpu_power,
            ram=self._ram_power
        )

    # %% power tracking methods
    def start(self):
        self._start = time.time()

    def record(self):
        self._step = time.time() - self._start
        self.record_cpu_power()
        self.record_gpu_power()
        self.record_ram_power()
        self.update_energy()

    def record_cpu_power(self):
        # TODO: find how to compute cpu power given percent, memory, etc.
        pass

    def record_gpu_power(self):
        r"""Record the current GPU power usage in watts
        WARNING: nvml returns the usage in milliwatts
        """
        if self._cuda:
            self._gpu_power = sum(nvidia_smi.nvmlDeviceGetPowerUsage(handle) for handle in self._handles) / 1000

    def record_ram_power(self):
        # TODO: same as CPU
        pass

    def update_energy(self):
        r"""Update the consumption energy in watt-hours
        """
        self.total_energy += self._step * (self._cpu_power + self._gpu_power + self._ram_power) / 3600
