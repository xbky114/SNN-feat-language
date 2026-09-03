import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt


class IF(nn.Module):
    def __init__(self, num_neurons: int, threshold: float):
        super().__init__()
        self.num_neurons = num_neurons
        self.threshold = threshold
        self.membrane_potential = None

    def forward(self, input_current: torch.Tensor):
        # lazy initialization of membrane potential
        if self.membrane_potential is None:
            batch_size = input_current.size(0)
            self.membrane_potential = torch.zeros(batch_size, self.num_neurons, device=input_current.device)
        
        # Update membrane potential based on input current
        self.membrane_potential += input_current

        # Generate spikes based on threshold
        spikes = (self.membrane_potential >= self.threshold).float()

        # Reset membrane potential for neurons that spiked
        self.membrane_potential[spikes.bool()] -= self.threshold

        return spikes

    def reset(self):
        self.membrane_potential = None

class simple_SNN(nn.Module):
    def __init__(self, num_neurons: int, threshold: float, T: int):
        super().__init__()
        self.if_layer = IF(num_neurons, threshold)
        self.T = T

    def reset(self):
        self.if_layer.reset()

    def forward(self, input_currents: torch.Tensor):
        self.reset()
        spikes = []
        # input_currents: [t, batch, dim]
        for t in range(self.T):
            input_current = input_currents[t]
            spike = self.if_layer(input_current)
            spikes.append(spike)
        spikes = torch.stack(spikes, dim=0)
        return spikes

    def average_spike_rate(self, spikes: torch.Tensor) -> np.ndarray:
        # spikes: [t, batch, dim]
        # Calculate the average spike rate for each time step
        accumulated_spikes = torch.cumsum(spikes, dim=0)
        num_neurons = spikes.size(2)
        return np.array([accumulated_spikes[t].sum().item() / num_neurons / (t + 1) for t in range(self.T)])

class GaussianData(Dataset):
    def __init__(self, num_samples: int, dim: int, mean: float = 0.0, std: float = 1.0):
        self.num_samples = num_samples
        self.dim = dim
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a sample of input currents from a Gaussian distribution
        input_current = torch.normal(mean=self.mean, std=self.std, size=(self.dim,))
        return input_current

def spike_rate_expectation(dataloader: DataLoader, num_neurons: int = 10, threshold: float = 1.0, T: int = 100, device: str = 'cpu') -> np.ndarray:
    dim = dataloader.dataset.dim
    num_samples = len(dataloader.dataset)
    assert num_neurons == dim

    # Create the SNN model
    snn_model = simple_SNN(num_neurons=num_neurons, threshold=threshold, T=T)
    snn_model.to(device)

    # Store average spike rates for each sample
    average_spike_rates = np.zeros((num_samples, T))

    for i, input_current in enumerate(dataloader):
        input_current = input_current.squeeze(0).to(device)  # Remove batch dimension and move to device
        spikes = snn_model(torch.repeat_interleave(input_current, T, dim=0))  # Repeat input for T time steps
        avg_spike_rate = snn_model.average_spike_rate(spikes)
        average_spike_rates[i] = avg_spike_rate

    spike_rate_expectation = np.mean(average_spike_rates, axis=0)

    return spike_rate_expectation

def plot_expected_spike_rate(spike_rate_expectation: np.ndarray, fit_curve: np.ndarray = None):
    plt.figure(figsize=(10, 6))
    plt.plot(spike_rate_expectation, color = 'blue', label='real data')
    if fit_curve is not None:
        plt.plot(fit_curve, color = 'red', label='fitted data', alpha = 0.5)
    plt.xlabel('Time Steps')
    plt.ylabel('Expected Spike Rate')
    plt.title('Spike Rate Expectation Over Time')
    plt.legend()
    plt.grid()
    plt.savefig('./expected_spike_rate.png')
    plt.close()

def fit_spike_rate_expectation(spike_rate_expectation: np.ndarray, abort_before: int = 10):
    # Fit the spike rate expectation to an exponential decay function
    from scipy.optimize import curve_fit

    def one_over_T(t, a, b):
        return b - a / t

    observed = spike_rate_expectation[abort_before:]

    t = np.arange(len(spike_rate_expectation) - abort_before) + abort_before + 1
    popt, _ = curve_fit(one_over_T, t, observed, p0=(0.1, 0.03))
   
    fitted = one_over_T(t, *popt)
    ss_res = np.sum((observed - fitted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return popt, r_squared

if __name__ == "__main__":
    device = "cuda:0"
    dataloader = DataLoader(GaussianData(num_samples=10000, dim=100, mean = 0.0, std = 0.2), batch_size=256, shuffle=True)
    spike_rate_expectation_result = spike_rate_expectation(dataloader, num_neurons=100, threshold=1.0, T=512, device=device)

    abort_before = 10
    fitted_params, r_squared = fit_spike_rate_expectation(spike_rate_expectation_result, abort_before=abort_before)
    fitted_data = [None] * abort_before + [fitted_params[1] - fitted_params[0] / (t + 1) for t in range(abort_before, len(spike_rate_expectation_result))]
    plot_expected_spike_rate(spike_rate_expectation_result, fit_curve=fitted_data)

    with open('./fitted_params.txt', 'w') as f:
        f.write(f"A_T = A_\infty - C / T\\\n")
        f.write(f"Fitted parameters: C = {fitted_params[0]}, A_\infty = {fitted_params[1]}\n")
        f.write(f"R^2 = {r_squared}\n")

