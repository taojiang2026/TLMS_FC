# TLMS Experiment Code

Experimental code accompanying the paper:

**A Low-Complexity Tensor-Based Acoustic Feedback Canceller for Hearing Aids with Steady-State Analysis**

## Run

Use Python 3.10 or later. From this directory:

```bash
python -m pip install -r requirements.txt
python "MIS comparison under different system gains.py"
```

The script loads `data/experiment_workspace.npz` and saves the figures and numerical results in `outputs/`.

- `fig2_noise.png`: white-noise input.
- `fig3_speech.png`: speech input.

These filenames are retained from the original experiments; the plots correspond to Figs. 3 and 4 in the revised manuscript.

## Files

- `experiment.py`: TLMS implementation.
- `MIS comparison under different system gains.py`: experiment and plotting script.
- `data/experiment_workspace.npz`: saved numerical inputs.
- `IMPLEMENTATION_NOTES.md`: implementation details and correspondence with the paper.

Run the numerical checks with:

```bash
python -m unittest discover -s tests -v
```
