import time

import torch
if torch.cuda.is_available():
    import nvidia_smi


class CO2Tracker:
    def __init__(self):
        r"""CO2 consumption tracker for deep learning models.
        Look at https://arxiv.org/abs/1906.02243 for details.
        """
        # temporal variables
        self._start = None
        self._step = None
        self._num_records = 0

        # power variables
        self._total_cpu_power = 0
        self._total_gpu_power = 0
        self._total_ram_power = 0

        # GPU-specific constants
        self._cuda = torch.cuda.is_available()
        if self._cuda:
            nvidia_smi.nvmlInit()
            self._handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

        # energy consumption constants
        self._pue_coeff = 1.58
        self._co2_coeff = 0.477

    @property
    def avg_cpu_power(self):
        return self._total_cpu_power / self._num_records

    @property
    def avg_gpu_power(self):
        return self._total_gpu_power / self._num_records

    @property
    def avg_ram_power(self):
        return self._total_ram_power / self._num_records

    @property
    def total_energy(self):
        return self._pue_coeff * self._step * (self.avg_cpu_power + self.avg_gpu_power + self.avg_ram_power) / 1000

    @property
    def co2_equivalent(self):
        return self._co2_coeff * self.total_energy

    def power_stats(self):
        return dict(
            cpu=self.avg_cpu_power,
            gpu=self.avg_gpu_power,
            ram=self.avg_ram_power
        )

    # %% power tracking methods
    def start(self):
        self._start = time.time()
        if self._cuda:
            nvidia_smi.nvmlInit()

    def record(self):
        self._step = time.time() - self._start
        self._num_records += 1
        self.record_cpu_power()
        self.record_gpu_power()
        self.record_ram_power()

    def record_cpu_power(self):
        # TODO: find how to compute cpu power given percent, memory, etc.
        self._total_cpu_power += 0  # dummy example

    def record_gpu_power(self):
        # TODO: test model on GPU
        if self._cuda:
            self._total_gpu_power += nvidia_smi.nvmlDeviceGetPowerUsage(self._handle)

    def record_ram_power(self):
        # TODO: same as CPU
        self._total_ram_power += 0  # dummy example
