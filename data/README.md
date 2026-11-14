# Numerical workspace

`experiment_workspace.npz` is the only required input file. It loads with `np.load(path, allow_pickle=False)`.

| Field | Shape | Description |
| --- | --- | --- |
| `feedback_path` | (256,) | Preprocessed measured feedback coefficients |
| `input_noise` | (20000,) | Peak-normalized white-noise input |
| `input_speech` | (20000,) | Normalized speech samples 60000 through 79999 |
| `microphone_noise_fig2` | (3, 20000) | Noise for gains 6, 12, 18 dB with white-noise input |
| `microphone_noise_fig3` | (3, 20000) | Noise for gains 6, 12, 18 dB with speech input |
| `gains_db` | (3,) | [6, 12, 18] |
| `sample_rate`, `seed`, `format_version` | scalar | 16000, 42, 1 |

The file contains the actual speech segment and feedback-path values. Their original redistribution terms still apply. No original WAV or CSV files are included in this folder.
