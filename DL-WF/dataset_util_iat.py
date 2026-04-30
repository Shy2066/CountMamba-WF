import numpy as np

class IATLogScaleHistogramDataset:
    def __init__(self, time_slots, log_bins):
        self.time_slots = time_slots
        self.log_bins = log_bins
        self.data = []

    def add_sample(self, sample):
        self.data.append(sample)

    def compute_iat(self):
        if len(self.data) <= 1:
            return np.zeros(len(self.log_bins) - 1)
        # Calculate inter-arrival times
        iat = np.diff(self.data)
        return iat

    def histogram(self):
        iat = self.compute_iat()
        # Logarithmic binning
        hist, _ = np.histogram(iat, bins=self.log_bins)
        return hist

    def get_fixed_time_slots(self):
        return np.linspace(0, max(self.data), num=self.time_slots)
