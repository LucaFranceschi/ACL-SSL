"""
visualize_heatmaps.py

Generates a grid plot of overlaid.jpg (or any image type) across experiments and samples.

Rows    = experiments (defined in EXP_DUMPS_PATH)
Columns = sample directories found under each experiment's best/ folder

Click any cell -> popup with all experiment images for that sample + audio player.

Dependencies:
    pip install pygame pillow

Usage:
    python visualize_heatmaps.py --mode best
    python visualize_heatmaps.py --root /path/to/train_outputs --img overlaid.jpg --mode worst --out grid.png
"""

import os
import time
import argparse
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Try to import pygame for audio; warn gracefully if missing
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    AUDIO_OK = True
except ImportError:
    AUDIO_OK = False
    print("[warn] pygame not found — audio disabled. Run: pip install pygame")

# ---------------------------------------------------------------------------
# Configuration — edit these to match your setup
# ---------------------------------------------------------------------------

ROOT = "train_outputs"   # path to the train_outputs directory

EXP_DUMPS_PATH = {
    'ACL_baseline': '2223542/Visual_results_test/vggss/ACL_ViT16_Exp_ACL_v1/epochbest',
    'ACL_v1_B16':   '2074301-full/2223543/Visual_results_test/vggss/ACL_ViT16_aclifa_2gpu/epoch17',
    'ACL_v1_B32':   'pirineus3/2064866/2223546/Visual_results_test/vggss/ACL_ViT16_aclifa_2gpu/epoch19',
    'ACL_v2_B16':   'pirineus3/2168632/2223544/Visual_results_test/vggss/ACL_ViT16_aclifa_2gpu/epoch16',
    'ACL_v3_B16':   'pirineus3/2210849/2223545/Visual_results_test/vggss/ACL_ViT16_aclifa_2gpu/epoch15',
    'ACL_v4_B16':   'pirineus3/2271991/2223547/Visual_results_test/vggss/ACL_ViT16_aclifa_2gpu/epoch18',
    'ACL_v5_B16':   'pirineus3/2568854/2223548/Visual_results_test/vggss/ACL_ViT16_aclifa_2gpu/epoch16',
}

IMAGE_FILENAME = "overlaid.jpg"   # which image to show in each cell

# ---------------------------------------------------------------------------
# Helpers (your original logic, untouched)
# ---------------------------------------------------------------------------

def collect_samples(dir: str) -> list[str]:
    """Return sorted list of sample subdirectory names."""
    return sorted([d for d in os.listdir(dir)])


def build_grid(root, mode):
    exp_names  = list(EXP_DUMPS_PATH.keys())
    sample_set = set()
    ret_dict   = {}

    for exp, dumps_rel in EXP_DUMPS_PATH.items():
        samples = collect_samples(os.path.join(root, dumps_rel, mode))
        if len(samples) > 0:
            ret_dict[exp] = os.path.join(root, dumps_rel, mode)
        sample_set.update(samples)
        print(f"  -> {dumps_rel}  ({len(samples)} samples)")

    all_samples = sorted(sample_set)
    return exp_names, ret_dict, all_samples


def load_image_or_placeholder(path: str):
    """Load image array, or return a grey placeholder if missing."""
    if path and os.path.isfile(path):
        try:
            return mpimg.imread(path), True
        except Exception:
            pass
    return np.full((90, 120, 3), 0.85), False

# ---------------------------------------------------------------------------
# Audio player (pygame-based)
# ---------------------------------------------------------------------------

class AudioPlayer:
    def __init__(self, wav_path):
        self.wav_path = wav_path
        self.duration = 0.0
        self.playing  = False
        self._start_t = 0.0
        self._offset  = 0.0
        self._loaded  = False
        self._load()

    def _load(self):
        if not AUDIO_OK or not self.wav_path or not os.path.isfile(self.wav_path):
            return
        try:
            pygame.mixer.music.load(self.wav_path)
            import wave, contextlib
            with contextlib.closing(wave.open(self.wav_path, 'r')) as f:
                self.duration = f.getnframes() / float(f.getframerate())
            self._loaded = True
        except Exception as e:
            print(f"  [audio] Could not load {self.wav_path}: {e}")

    def play(self):
        if not self._loaded: return
        pygame.mixer.music.play(start=self._offset)
        self._start_t = time.time()
        self.playing  = True

    def pause(self):
        if not self._loaded or not self.playing: return
        self._offset += time.time() - self._start_t
        pygame.mixer.music.pause()
        self.playing = False

    def seek(self, seconds):
        if not self._loaded: return
        self._offset = max(0.0, min(seconds, self.duration))
        if self.playing:
            pygame.mixer.music.play(start=self._offset)
            self._start_t = time.time()

    def position(self):
        if not self._loaded: return 0.0
        if self.playing:
            return min(self._offset + (time.time() - self._start_t), self.duration)
        return self._offset

    def stop(self):
        if self._loaded: pygame.mixer.music.stop()
        self.playing = False
        self._offset = 0.0

# ---------------------------------------------------------------------------
# Popup: all experiment images for one sample + audio player
# ---------------------------------------------------------------------------

POPUP_WIN_W = 1080   # default popup window width  — change this
POPUP_WIN_H = 800   # default popup window height — change this

# Image size inside each row: leave room for the experiment label (~140px) and padding
POPUP_IMG_W = 900 - 160
POPUP_IMG_H = int(POPUP_IMG_W * 0.75)  # keep 4:3 aspect ratio

_current_popup = {"win": None, "player": None}

def open_popup(sample_id, exp_names, best_dirs, img_filename, audio_path):
    # Close any existing popup
    prev = _current_popup["win"]
    if prev is not None:
        try:
            _current_popup["player"].stop()
            prev.destroy()
        except Exception:
            pass
        _current_popup["win"] = None
        _current_popup["player"] = None

    win = tk.Toplevel()
    win.title(f"Sample: {sample_id}")
    win.configure(bg="#1e1e1e")
    win.resizable(True, True)
    win.geometry(f"{POPUP_WIN_W}x{POPUP_WIN_H}")

    tk.Label(win, text=sample_id, font=("Helvetica", 13, "bold"),
             bg="#1e1e1e", fg="#eeeeee").pack(pady=(10, 4))

    # Scrollable image area
    frame_outer = tk.Frame(win, bg="#1e1e1e")
    frame_outer.pack(fill=tk.BOTH, expand=True, padx=10)

    canvas_scroll = tk.Canvas(frame_outer, bg="#1e1e1e", highlightthickness=0)
    scrollbar = ttk.Scrollbar(frame_outer, orient="vertical", command=canvas_scroll.yview)
    canvas_scroll.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    inner_frame = tk.Frame(canvas_scroll, bg="#1e1e1e")
    canvas_scroll.create_window((0, 0), window=inner_frame, anchor="nw")

    win._photo_refs = []  # attached to window to survive GC for its entire lifetime

    for exp in exp_names:
        best_dir = best_dirs.get(exp)
        img_path = os.path.join(best_dir, sample_id, img_filename) if best_dir else None

        row_frame = tk.Frame(inner_frame, bg="#2a2a2a", bd=1, relief=tk.FLAT)
        row_frame.pack(fill=tk.X, pady=3, padx=2)

        tk.Label(row_frame, text=exp, font=("Helvetica", 10, "bold"),
                 bg="#2a2a2a", fg="#aaaaff", width=16, anchor="w").pack(side=tk.LEFT, padx=8)

        if img_path and os.path.isfile(img_path):
            try:
                pil_img = Image.open(img_path).resize((POPUP_IMG_W, POPUP_IMG_H), Image.LANCZOS)
                photo   = ImageTk.PhotoImage(pil_img)
                win._photo_refs.append(photo)  # keep alive on the window, not a local var
                lbl = tk.Label(row_frame, image=photo, bg="#2a2a2a")
                lbl.image = photo  # double-anchor: also on the label itself
                lbl.pack(side=tk.LEFT, padx=4, pady=4)
            except Exception as e:
                tk.Label(row_frame, text=f"[error: {e}]",
                         bg="#2a2a2a", fg="#888888").pack(side=tk.LEFT, padx=8)
        else:
            tk.Label(row_frame, text="— not found —",
                     bg="#2a2a2a", fg="#666666").pack(side=tk.LEFT, padx=8, pady=POPUP_IMG_H // 2)

    inner_frame.update_idletasks()
    canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))

    # Mousewheel scroll
    canvas_scroll.bind("<MouseWheel>", lambda e: canvas_scroll.yview_scroll(int(-1*(e.delta/120)), "units"))
    canvas_scroll.bind("<Button-4>",   lambda e: canvas_scroll.yview_scroll(-1, "units"))
    canvas_scroll.bind("<Button-5>",   lambda e: canvas_scroll.yview_scroll( 1, "units"))

    # Audio player
    audio_frame = tk.Frame(win, bg="#141414", pady=8)
    audio_frame.pack(fill=tk.X, padx=10, pady=(6, 10))

    tk.Label(audio_frame, text="🔊  Audio", font=("Helvetica", 10, "bold"),
             bg="#141414", fg="#cccccc").grid(row=0, column=0, padx=8, sticky="w")

    player = AudioPlayer(audio_path)
    _current_popup["win"]    = win
    _current_popup["player"] = player
    time_var = tk.StringVar(value="0:00 / 0:00")
    tk.Label(audio_frame, textvariable=time_var, font=("Courier", 9),
             bg="#141414", fg="#aaaaaa").grid(row=0, column=2, padx=8)

    seek_var = tk.DoubleVar(value=0.0)
    seek_bar = ttk.Scale(audio_frame, from_=0, to=max(player.duration, 1.0),
                         orient=tk.HORIZONTAL, variable=seek_var, length=POPUP_IMG_W * 2)
    seek_bar.grid(row=1, column=0, columnspan=4, padx=8, pady=4, sticky="ew")
    audio_frame.columnconfigure(1, weight=1)

    _seeking = [False]
    seek_bar.bind("<ButtonPress-1>",   lambda e: _seeking.__setitem__(0, True))
    seek_bar.bind("<ButtonRelease-1>", lambda e: (player.seek(seek_var.get()), _seeking.__setitem__(0, False)))

    btn_var = tk.StringVar(value="▶  Play")

    def toggle_play():
        if player.playing:
            player.pause(); btn_var.set("▶  Play")
        else:
            player.play();  btn_var.set("⏸  Pause")

    audio_available = AUDIO_OK and bool(audio_path) and os.path.isfile(audio_path or "")
    tk.Button(audio_frame, textvariable=btn_var, font=("Helvetica", 10),
              command=toggle_play, bg="#3a3a5c", fg="#ffffff",
              activebackground="#5555aa", relief=tk.FLAT, padx=12,
              state=tk.NORMAL if audio_available else tk.DISABLED,
              ).grid(row=0, column=1, padx=8, sticky="w")

    if not audio_available:
        msg = "(pygame not installed)" if not AUDIO_OK else "(audio.wav not found)"
        tk.Label(audio_frame, text=msg, bg="#141414", fg="#666666",
                 font=("Helvetica", 8)).grid(row=2, column=0, columnspan=4, sticky="w", padx=8)

    def fmt(s): s = int(s); return f"{s//60}:{s%60:02d}"

    def poll():
        if not win.winfo_exists(): return
        pos = player.position()
        if not _seeking[0]: seek_var.set(pos)
        time_var.set(f"{fmt(pos)} / {fmt(player.duration)}")
        if player.playing and pos >= player.duration - 0.1:
            player.stop(); btn_var.set("▶  Play")
        win.after(200, poll)

    poll()
    def on_close():
        player.stop()
        _current_popup["win"]    = None
        _current_popup["player"] = None
        win.destroy()
    win.protocol("WM_DELETE_WINDOW", on_close)
    win.lift()
    win.focus_force()

# ---------------------------------------------------------------------------
# Main plotting logic (your original, with click handler added)
# ---------------------------------------------------------------------------

def plot_grid(root: str, img_filename: str, output_path: str, mode,
              max_cols: int | None = None, dpi: int = 150):

    exp_names, best_dirs, all_samples = build_grid(root, mode)

    if not all_samples:
        print("No samples found — check your ROOT path and directory structure.")
        return

    if max_cols:
        all_samples = all_samples[:max_cols]

    n_exp  = len(exp_names)
    n_cols = len(all_samples)

    # --- Row heights (in inches) ---
    cell_w,  cell_h  = 1.8, 1.4   # experiment rows
    label_w           = 1.6
    top_h             = 0.6

    # Header row 1: original_frame — same height as experiment rows
    frame_row_h = cell_h

    # Header row 2: waveform — probe the first available waveform to get its aspect ratio
    waveform_aspect = None   # width / height
    for sample in all_samples:
        for e in exp_names:
            bd = best_dirs.get(e)
            if bd:
                wp = os.path.join(bd, sample, "waveform.jpg")
                if os.path.isfile(wp):
                    try:
                        im = Image.open(wp)
                        waveform_aspect = im.width / im.height
                    except Exception:
                        pass
                    break
        if waveform_aspect is not None:
            break
    if waveform_aspect is None:
        waveform_aspect = 4.0   # fallback if no waveform found
    wave_row_h = cell_w / waveform_aspect   # height that preserves ratio given cell_w

    n_rows_total = 2 + n_exp   # waveform + frame + experiments

    row_heights = [frame_row_h, wave_row_h] + [cell_h] * n_exp

    fig_w = label_w + n_cols * cell_w
    fig_h = top_h   + sum(row_heights)

    fig, axes = plt.subplots(
        n_rows_total, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        gridspec_kw={"height_ratios": row_heights},
    )

    fig.patch.set_facecolor("#f7f7f7")
    fig.canvas.manager.set_window_title("Heatmap grid — click a cell to expand + play audio")

    axes_map = {}   # ax -> (sample_id, audio_path)

    # ------------------------------------------------------------------
    # Row 0 — original_frame.jpg  (stretch, no aspect ratio kept)
    # ------------------------------------------------------------------
    for c, sample in enumerate(all_samples):
        ax = axes[0, c]
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc"); spine.set_linewidth(0.4)

        img_path = None
        for e in exp_names:
            bd = best_dirs.get(e)
            if bd:
                p = os.path.join(bd, sample, "original_frame.jpg")
                if os.path.isfile(p):
                    img_path = p; break

        img, found = load_image_or_placeholder(img_path)
        ax.imshow(img, aspect='auto')   # stretch to fill
        if not found:
            ax.text(0.5, 0.5, "—", transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color="#aaaaaa")

    axes[0, 0].set_ylabel(
        "Input\nFrame", fontsize=8, fontweight='bold', rotation=0,
        labelpad=4, ha='right', va='center', color="#222222"
    )

    # Row 1 — waveform.jpg  (keep aspect ratio via letterboxing, full cell width)
    # ------------------------------------------------------------------
    # Compute the target aspect ratio for a waveform cell (width / height)
    target_aspect = cell_w / wave_row_h

    for c, sample in enumerate(all_samples):
        ax = axes[1, c]
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc"); spine.set_linewidth(0.4)

        img_path = None
        for e in exp_names:
            bd = best_dirs.get(e)
            if bd:
                p = os.path.join(bd, sample, "waveform.jpg")
                if os.path.isfile(p):
                    img_path = p; break

        img, found = load_image_or_placeholder(img_path)

        if found:
            # Letterbox: embed the image centred in a canvas of exactly target_aspect
            h, w = img.shape[:2]
            img_aspect = w / h
            if img_aspect > target_aspect:
                # Image is wider than cell → scale to full width, add top/bottom padding
                new_w = w
                new_h = int(round(w / target_aspect))
            else:
                # Image is taller than cell → scale to full height, add left/right padding
                new_h = h
                new_w = int(round(h * target_aspect))
            canvas = np.full((new_h, new_w, 3), 0.97)   # near-white background
            y0 = (new_h - h) // 2
            x0 = (new_w - w) // 2
            canvas[y0:y0+h, x0:x0+w] = img if img.max() <= 1.0 else img / 255.0
            img = canvas

        ax.imshow(img, aspect='auto')   # now fills the full cell
        if not found:
            ax.text(0.5, 0.5, "—", transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color="#aaaaaa")

    axes[1, 0].set_ylabel(
        "Input\nWaveform", fontsize=8, fontweight='bold', rotation=0,
        labelpad=4, ha='right', va='center', color="#222222"
    )

    # ------------------------------------------------------------------
    # Rows 2..  — experiment heatmaps  (unchanged logic)
    # ------------------------------------------------------------------
    for r, exp in enumerate(exp_names):
        row_idx  = r + 2
        best_dir = best_dirs.get(exp)

        for c, sample in enumerate(all_samples):
            ax = axes[row_idx, c]
            ax.set_xticks([]); ax.set_yticks([])

            img_path = os.path.join(best_dir, sample, img_filename) if best_dir else None
            img, found = load_image_or_placeholder(img_path)
            ax.imshow(img, aspect='auto')

            for spine in ax.spines.values():
                spine.set_edgecolor("#cccccc"); spine.set_linewidth(0.4)

            if not found:
                ax.text(0.5, 0.5, "—", transform=ax.transAxes,
                        ha='center', va='center', fontsize=10, color="#aaaaaa")

            audio_path = None
            for e in exp_names:
                bd = best_dirs.get(e)
                if bd:
                    ap = os.path.join(bd, sample, "audio.wav")
                    if os.path.isfile(ap):
                        audio_path = ap; break

            axes_map[ax] = (sample, audio_path)

        axes[row_idx, 0].set_ylabel(
            exp, fontsize=8, fontweight='bold', rotation=0,
            labelpad=4, ha='right', va='center', color="#222222"
        )

    fig.subplots_adjust(
        left=label_w / fig_w,
        right=0.99,
        top=1.0 - top_h / fig_h,
        bottom=0.01,
        wspace=0.04,
        hspace=0.08,
    )

    fig.suptitle(
        f"Qualitative results: ACL-SaN model comparison of overlaid heatmaps",
        x=(label_w / fig_w + 0.99) / 2,
        y=1.0 - (top_h / fig_h) / 2,
        fontsize=11, fontweight='bold', color="#111111",
        va='center',
    )

    def on_click(event):
        if event.inaxes is None or event.inaxes not in axes_map:
            return
        sample_id, audio_path = axes_map[event.inaxes]
        open_popup(sample_id, exp_names, best_dirs, img_filename, audio_path)

    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\nSaved -> {output_path}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot heatmap grid across experiments and samples")
    parser.add_argument("--root",     default=ROOT,               help="Path to train_outputs directory")
    parser.add_argument("--img",      default=IMAGE_FILENAME,     help="Image filename to display (default: overlaid.jpg)")
    parser.add_argument("--out",      default="heatmap_grid.png", help="Output PNG path")
    parser.add_argument("--max-cols", type=int, default=None,     help="Limit number of sample columns shown")
    parser.add_argument("--dpi",      type=int, default=300,      help="Output DPI (default: 300)")
    parser.add_argument(
        "--mode",
        choices=["best", "worst"],
        required=True,
        help="Whether to process best or worst samples"
    )
    args = parser.parse_args()

    plot_grid(
        root=args.root,
        img_filename=args.img,
        output_path=args.out,
        mode=args.mode,
        max_cols=args.max_cols,
        dpi=args.dpi,
    )