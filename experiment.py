"""Norm-normalized TLMS and offline system-gain experiments."""
import numpy as np
from scipy.signal import lfilter

class Config:
    def __init__(self):
        self.fs = 16000
        self.fb_length = 256
        self.filter_length = 256


class TLMS_V3:
    """Two-factor TLMS. Both factors use the same pre-update error."""

    def __init__(self, length, mu_a=0.1, mu_b=0.005, size_a=16, size_b=16,
                 use_decay=False, use_norm_constraint=False,
                 decay_rate=0.99995, max_norm=5.0):
        dimensions = (length, size_a, size_b)
        if any(not isinstance(v, (int, np.integer)) or v <= 0 for v in dimensions):
            raise ValueError('Filter dimensions must be positive integers.')
        if size_a * size_b != length:
            raise ValueError('size_a * size_b must equal length.')
        if not all(np.isfinite(v) and v >= 0 for v in (mu_a, mu_b)):
            raise ValueError('Step sizes must be finite and nonnegative.')
        if use_decay and not (0 < decay_rate <= 1):
            raise ValueError('decay_rate must lie in (0, 1].')
        if use_norm_constraint and not (np.isfinite(max_norm) and max_norm > 0):
            raise ValueError('max_norm must be finite and positive.')
        self.length = length
        self.size_a = size_a
        self.size_b = size_b
        self.mu_a = mu_a
        self.mu_b = mu_b
        self.use_decay = use_decay
        self.use_norm_constraint = use_norm_constraint
        self.decay_rate = decay_rate
        self.max_norm = max_norm
        self.errors = []
        self.misalignment = []

    def run(self, x, d, h_true=None):
        """Run a fresh adaptation and return the online error signal."""
        x = np.asarray(x, dtype=float)
        d = np.asarray(d, dtype=float)
        if x.ndim != 1 or d.shape != x.shape or len(x) == 0:
            raise ValueError('x and d must be nonempty one-dimensional arrays of equal length.')
        if not np.isfinite(x).all() or not np.isfinite(d).all():
            raise ValueError('x and d must be finite.')
        if h_true is not None:
            h_true = np.asarray(h_true, dtype=float)
            if h_true.shape != (self.length,) or not np.isfinite(h_true).all():
                raise ValueError('h_true must be finite and match the filter length.')
        self.errors = []
        self.misalignment = []
        M_ha, M_hb = self.size_a, self.size_b
        n = len(x)

        h_a = np.ones((M_ha, 1)) * 0.001
        h_b = np.ones((M_hb, 1)) * 0.001

        e = np.zeros(n)
        x_pad = np.pad(x, (self.length - 1, 0), 'constant')

        for k in range(n):
            x_k = x_pad[k:k + self.length][::-1]
            # Column-major order matches h = kron(h_b, h_a).
            X_k = x_k.reshape((M_ha, M_hb), order='F')

            # Reuse X_k @ h_b in the output and the update.
            z_k = np.dot(X_k, h_b)
            y_out = np.dot(h_a.T, z_k)[0, 0]
            e[k] = d[k] - y_out

            y_k = np.dot(X_k.T, h_a)

            if self.use_decay:
                current_mu_a = self.mu_a * (self.decay_rate ** k)
                current_mu_b = self.mu_b * (self.decay_rate ** k)
            else:
                current_mu_a = self.mu_a
                current_mu_b = self.mu_b

            # Divide by the L2 norm, not its square (Algorithm 1).
            z_norm = np.linalg.norm(z_k) + 1e-8
            y_norm = np.linalg.norm(y_k) + 1e-8

            delta_h_a = current_mu_a * e[k] * z_k / z_norm
            delta_h_b = current_mu_b * e[k] * y_k / y_norm

            # Both directions are computed before either factor changes.
            h_a = h_a + delta_h_a
            h_b = h_b + delta_h_b

            if self.use_norm_constraint:
                total_norm = np.sqrt(np.sum(h_a**2) + np.sum(h_b**2))
                if total_norm > self.max_norm:
                    scale = self.max_norm / total_norm
                    h_a *= scale
                    h_b *= scale

            self.errors.append(e[k]**2)

            if h_true is not None and k == 0:
                h_est = np.kron(h_b.flatten(), h_a.flatten())
                mis = np.linalg.norm(h_est[:len(h_true)] - h_true) / (np.linalg.norm(h_true) + 1e-10)
                self.misalignment.append((k, mis))
            elif h_true is not None and k >= 64 and k % 100 == 0:
                h_est = np.kron(h_b.flatten(), h_a.flatten())
                mis = np.linalg.norm(h_est[:len(h_true)] - h_true) / (np.linalg.norm(h_true) + 1e-10)
                self.misalignment.append((k, mis))

        self.h_a = h_a
        self.h_b = h_b
        return e

    def get_weights(self):
        return np.kron(self.h_b.flatten(), self.h_a.flatten())


def smooth_ema(data, alpha=0.99):
    """Smooth a series with an exponential moving average."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 1 or not (0 <= alpha < 1):
        raise ValueError('Use a one-dimensional series and alpha in [0, 1).')
    if len(data) == 0:
        return data.copy()
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * data[i]
    return smoothed


def calculate_erle(echo_signal, residual):
    """Measure feedback attenuation using the final filter coefficients."""
    psignal = np.mean(echo_signal ** 2) + 1e-10
    presidual = np.mean(residual ** 2) + 1e-10
    return 10 * np.log10(psignal / presidual)


def run_single_experiment(x_original, h_feedback, G, noise_std, fs, cfg, algo_params, noise=None):
    """Run both step-size settings with the same input and microphone noise."""
    x_original = np.asarray(x_original, dtype=float)
    h_feedback = np.asarray(h_feedback, dtype=float)
    if x_original.ndim != 1 or len(x_original) == 0 or not np.isfinite(x_original).all():
        raise ValueError('x_original must be a finite nonempty one-dimensional array.')
    if h_feedback.shape != (cfg.filter_length,) or not np.isfinite(h_feedback).all():
        raise ValueError('h_feedback must be finite and match the filter length.')
    if not (np.isfinite(fs) and fs > 0 and np.isfinite(G) and G > 0):
        raise ValueError('Sample rate and gain must be finite and positive.')
    if not (np.isfinite(noise_std) and noise_std >= 0):
        raise ValueError('noise_std must be finite and nonnegative.')
    delay_samples = int(0.004 * fs)  # Fixed 4 ms delay.

    # Offline receiver input: delayed and amplified clean input.
    u = np.zeros(len(x_original))
    if delay_samples < len(x_original):
        u[delay_samples:] = x_original[:-delay_samples] * G
    else:
        u[:] = 0

    # Feedback signal from the measured path.
    echo_signal = lfilter(h_feedback, [1], u)

    # Microphone signal: direct input, feedback, and noise.
    if noise is None:
        noise = np.random.randn(len(x_original)) * noise_std
    else:
        noise = np.asarray(noise, dtype=float)
        if noise.shape != x_original.shape or not np.isfinite(noise).all():
            raise ValueError('noise must be finite and match the input shape.')
    mic = x_original + echo_signal + noise

    # The two variants differ only in step size.
    algos = {
        'TLMS-1': lambda: TLMS_V3(cfg.filter_length, mu_a=algo_params['tlms1_mu_a'],
                                  mu_b=algo_params['tlms1_mu_b'], size_a=16, size_b=16,
                                  use_decay=False, use_norm_constraint=False),
        'TLMS-2': lambda: TLMS_V3(cfg.filter_length, mu_a=algo_params['tlms2_mu_a'],
                                  mu_b=algo_params['tlms2_mu_b'], size_a=16, size_b=16,
                                  use_decay=False, use_norm_constraint=False),
    }

    results = {}
    for name, algo_fn in algos.items():
        algo = algo_fn()
        e = algo.run(u, mic, h_feedback)
        corr = np.corrcoef(h_feedback, algo.get_weights()[:len(h_feedback)])[0, 1]
        mse = np.mean(algo.errors[-1000:])
        # ERLE uses the final filter, not the online microphone residual.
        y = lfilter(algo.get_weights(), [1], u)[:len(u)]
        residual = echo_signal - y
        erle = calculate_erle(echo_signal, residual)

        results[name] = {
            'filter': algo,
            'corr': corr,
            'mse': mse,
            'erle': erle,
            'e': e,
            'misalignment': algo.misalignment.copy() if hasattr(algo, 'misalignment') else []
        }

    return results, echo_signal
