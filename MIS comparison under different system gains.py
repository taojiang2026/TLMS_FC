"""Generate TLMS misalignment curves under different system gains."""
import argparse
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator
import numpy as np
import scipy

from experiment import Config, run_single_experiment, smooth_ema

ROOT = Path(__file__).resolve().parent
GAINS = (6, 12, 18)
NAMES = ('TLMS-1', 'TLMS-2')
PARAMS = dict(tlms1_mu_a=0.02, tlms1_mu_b=0.02, tlms2_mu_a=0.04, tlms2_mu_b=0.04)
COLORS = ('#e74c3c', '#f39c12')
STYLES = (':', '-')
MARKERS = ('D', 'p')
# Axis ranges from the original five-curve figures with the default seed (42).
# Keep their scale when displaying only the two TLMS settings.
Y_LIMITS = {
    2: {6: (-17.723507874694054, 0.8460762900685613),
        12: (-19.086848761044163, 0.9094331831302539),
        18: (-20.082133090857063, 0.9589161838128155)},
    3: {6: (-6.751370376284568, 0.3277071515592808),
        12: (-7.503549097624836, 0.35863025303891577),
        18: (-7.485046060832626, 0.3567991588016956)},
}


def draw_panel(ax, results, figure_number, gain):
    """Use the original MIS smoothing, decibel conversion, and line styles."""
    for name, color, style, marker in zip(NAMES, COLORS, STYLES, MARKERS):
        curve = np.asarray(results[name]['misalignment'])
        mis_db = 20 * np.log10(smooth_ema(curve[:, 1], alpha=0.98) + 1e-10)
        ax.plot(curve[:, 0], mis_db, color=color, linestyle=style,
                linewidth=2, label=name, marker=marker, markersize=5,
                markevery=max(1, len(curve) // 8), alpha=0.9)
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('MIS (dB)', fontsize=11)
    ax.set_xlim(0, 20000)
    ax.xaxis.set_major_locator(MultipleLocator(2500))
    ax.set_ylim(*Y_LIMITS[figure_number][gain])
    ax.yaxis.set_major_locator(MultipleLocator(2.5 if figure_number == 2 else 1))
    ax.legend(loc='lower left', frameon=True, fancybox=False,
              edgecolor='black', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', pad=2)


def run_figure(number, workspace, output_dir, seed):
    cfg = Config()
    h = workspace['feedback_path']
    if h.shape != (256,) or not np.isfinite(h).all() or np.linalg.norm(h) == 0:
        raise ValueError('Feedback path must contain a finite, nonzero, one-dimensional response.')
    kind = 'noise' if number == 2 else 'speech'
    x = workspace[f'input_{kind}']
    noises = workspace[f'microphone_noise_fig{number}']
    if seed != int(workspace['seed']):
        # Custom seeds change microphone noise; cached input signals stay fixed.
        noises = np.random.RandomState(seed).randn(3, 20000) * 0.001
    if x.shape != (20000,) or not np.isfinite(x).all() or np.max(np.abs(x)) == 0:
        raise ValueError('The experiment input must be finite and non-silent.')
    if noises.shape != (3, 20000) or not np.isfinite(noises).all():
        raise ValueError('Workspace noise must be finite and have shape (3, 20000).')

    combined, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    metrics, arrays = {}, {}
    for panel, gain in enumerate(GAINS):
        print(f'Fig. {number}: {kind}, gain={gain} dB', flush=True)
        results, _ = run_single_experiment(x, h, 10 ** (gain / 20),
                                          0.001, cfg.fs, cfg, PARAMS, noise=noises[panel])
        metrics[str(gain)] = {}
        for name in NAMES:
            r = results[name]
            curve = np.asarray(r['misalignment'])
            if not np.isfinite(curve).all() or not np.isfinite(r['filter'].get_weights()).all():
                raise RuntimeError(f'Non-finite result: Fig. {number}, {gain} dB, {name}')
            arrays[f'gain{gain}_{name}_misalignment'] = curve
            arrays[f'gain{gain}_{name}_weights'] = r['filter'].get_weights()
            metrics[str(gain)][name] = {key: float(r[key]) for key in ('corr', 'mse', 'erle')}
        draw_panel(axes[panel], results, number, gain)
        axes[panel].set_title(f'({chr(97 + panel)}) Gain = {gain} dB')
        single, ax = plt.subplots(figsize=(5, 3.5))
        draw_panel(ax, results, number, gain)
        single.tight_layout(pad=0.3)
        single.savefig(output_dir / f'fig{number}{chr(97 + panel)}_{kind}_gain{gain}dB.png',
                       dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close(single)
    combined.tight_layout(pad=0.5)
    for extension in ('png', 'pdf'):
        combined.savefig(output_dir / f'fig{number}_{kind}.{extension}', dpi=300,
                         bbox_inches='tight', pad_inches=0.05)
    plt.close(combined)
    np.savez_compressed(output_dir / f'fig{number}_{kind}_results.npz', **arrays)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', choices=('2', '3', 'all'), default='all')
    parser.add_argument('--workspace', type=Path, default=ROOT / 'data' / 'experiment_workspace.npz')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error('--seed must be an integer in [0, 2**32).')
    figures = (2, 3) if args.figure == 'all' else (int(args.figure),)
    if not args.workspace.is_file():
        parser.error(f'Missing workspace: {args.workspace}. See data/README.md.')
    with np.load(args.workspace, allow_pickle=False) as saved:
        workspace = {key: saved[key] for key in saved.files}
    required = {'format_version', 'sample_rate', 'gains_db', 'seed', 'feedback_path'}
    for number in figures:
        required.update({f'input_{"noise" if number == 2 else "speech"}',
                         f'microphone_noise_fig{number}'})
    missing = required - workspace.keys()
    if missing:
        parser.error('Missing workspace fields: ' + ', '.join(sorted(missing)))
    if (int(workspace['format_version']) != 1 or int(workspace['sample_rate']) != 16000
            or not np.array_equal(workspace['gains_db'], GAINS)):
        parser.error('Workspace version, sample rate, or gains do not match this experiment.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({'font.family': 'Times New Roman' if 'Times New Roman' in fonts else 'DejaVu Serif',
                         'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12,
                         'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9})
    report = {'seed': args.seed, 'gains_db': GAINS, 'sample_rate': 16000,
              'samples': 20000, 'filter_length': 256, 'delay_ms': 4,
              'noise_std': 0.001, 'algorithm_parameters': PARAMS,
              'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                           'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__},
              'figures': {}}
    for number in figures:
        report['figures'][str(number)] = run_figure(number, workspace, args.output_dir, args.seed)
    (args.output_dir / f'metrics_{args.figure}.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Figures and numerical results saved in: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
