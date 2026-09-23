"""Sonoscope : enregistrer un son (produit) et le décomposer en fréquences (interface Tkinter)."""

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from sonoscope import analysis, audio_io


class SonoscopeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sonoscope")
        self.geometry("1100x950")

        self.data = None
        self.sample_rate = None
        self.reference_data = None
        self.reference_sample_rate = None
        self.reference_label_var = tk.StringVar(value="Aucune référence chargée")

        self._build_controls()
        self._build_comparison()
        self._build_plots()

    def _build_controls(self):
        frame = ttk.Frame(self, padding=10)
        frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(frame, text="Durée (s) :").pack(side=tk.LEFT)
        self.duration_var = tk.IntVar(value=5)
        ttk.Spinbox(frame, from_=1, to=30, textvariable=self.duration_var, width=5).pack(side=tk.LEFT, padx=(0, 10))

        self.record_btn = ttk.Button(frame, text="Enregistrer", command=self.on_record)
        self.record_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(frame, text="Importer un fichier...", command=self.on_import).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Écouter", command=self.on_play).pack(side=tk.LEFT, padx=5)

        ttk.Label(frame, text="Nom du produit :").pack(side=tk.LEFT, padx=(20, 5))
        self.product_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.product_var, width=20).pack(side=tk.LEFT)
        ttk.Button(frame, text="Sauvegarder", command=self.on_save).pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar(value="Enregistrez un son ou importez un fichier.")
        ttk.Label(self, textvariable=self.status_var, padding=(10, 0)).pack(side=tk.TOP, fill=tk.X)

    def _build_comparison(self):
        frame = ttk.LabelFrame(self, text="Comparaison à une référence", padding=(10, 5))
        frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 5))

        header = ttk.Frame(frame)
        header.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(header, text="Charger une référence...", command=self.on_load_reference).pack(side=tk.LEFT)
        ttk.Label(header, textvariable=self.reference_label_var, padding=(10, 0)).pack(side=tk.LEFT)

        columns = ("ref", "cur", "delta_hz", "delta_pct")
        headings = ("Référence (Hz)", "Actuel (Hz)", "Écart (Hz)", "Écart (%)")
        self.comparison_tree = ttk.Treeview(frame, columns=columns, show="headings", height=4)
        for col, label in zip(columns, headings, strict=True):
            self.comparison_tree.heading(col, text=label)
            self.comparison_tree.column(col, width=140, anchor=tk.CENTER)
        self.comparison_tree.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))

    def _build_plots(self):
        notebook = ttk.Notebook(self)
        notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig_wave, self.ax_wave, self.canvas_wave = self._make_plot_tab(notebook, "Forme d'onde")
        self.fig_fft, self.ax_fft, self.canvas_fft = self._make_plot_tab(notebook, "Spectre FFT")
        self.fig_spec, self.ax_spec, self.canvas_spec = self._make_plot_tab(notebook, "Spectrogramme")

    def _make_plot_tab(self, notebook, title):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=title)

        figure = Figure(figsize=(10, 6), dpi=100)
        ax = figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(figure, master=tab)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        return figure, ax, canvas

    def on_record(self):
        self.record_btn.config(state=tk.DISABLED)
        self.status_var.set(f"Enregistrement en cours ({self.duration_var.get()}s)...")
        threading.Thread(target=self._record_worker, daemon=True).start()

    def _record_worker(self):
        try:
            data, sr = audio_io.record_from_mic(self.duration_var.get())
        except Exception as exc:  # noqa: BLE001 - erreur micro remontée telle quelle à l'utilisateur
            self.after(0, lambda exc=exc: self._on_record_error(exc))
            return
        self.after(0, lambda: self._on_audio_ready(data, sr, "Enregistrement terminé."))

    def _on_record_error(self, exc):
        self.record_btn.config(state=tk.NORMAL)
        messagebox.showerror("Erreur micro", str(exc))
        self.status_var.set("Erreur lors de l'enregistrement.")

    def on_import(self):
        path = filedialog.askopenfilename(
            title="Choisir un fichier audio",
            filetypes=[("Fichiers audio", "*.wav *.flac *.ogg *.mp3"), ("Tous les fichiers", "*.*")],
        )
        if not path:
            return
        try:
            data, sr = audio_io.load_audio_file(path)
        except Exception as exc:  # noqa: BLE001 - erreur de lecture remontée telle quelle à l'utilisateur
            messagebox.showerror("Erreur d'import", str(exc))
            return
        self._on_audio_ready(data, sr, f"Fichier chargé : {path}")

    def _on_audio_ready(self, data, sr, message):
        self.record_btn.config(state=tk.NORMAL)
        self.data, self.sample_rate = data, sr
        self.status_var.set(message)
        self._plot()

    def on_load_reference(self):
        initialdir = "data/recordings" if Path("data/recordings").is_dir() else "."
        path = filedialog.askopenfilename(
            title="Choisir un enregistrement de référence",
            initialdir=initialdir,
            filetypes=[("Fichiers audio", "*.wav *.flac *.ogg *.mp3"), ("Tous les fichiers", "*.*")],
        )
        if not path:
            return
        try:
            data, sr = audio_io.load_audio_file(path)
        except Exception as exc:  # noqa: BLE001 - erreur de lecture remontée telle quelle à l'utilisateur
            messagebox.showerror("Erreur d'import", str(exc))
            return
        self.reference_data, self.reference_sample_rate = data, sr
        self.reference_label_var.set(f"Référence : {Path(path).name}")
        if self.data is not None:
            self._plot()

    def on_play(self):
        if self.data is None:
            return
        import sounddevice as sd

        sd.play(self.data, self.sample_rate)

    def on_save(self):
        if self.data is None:
            return
        path = audio_io.save_recording(self.data, self.sample_rate, self.product_var.get() or "produit")
        messagebox.showinfo("Sauvegardé", f"Enregistrement sauvegardé :\n{path}")

    def _plot(self):
        data, sr = self.data, self.sample_rate

        self.ax_wave.clear()
        t = np.linspace(0, len(data) / sr, len(data))
        self.ax_wave.plot(t, data, linewidth=0.8)
        self.ax_wave.set_title("Forme d'onde")
        self.ax_wave.set_xlabel("Temps (s)")
        self.ax_wave.set_ylabel("Amplitude")
        self.fig_wave.tight_layout(pad=3)
        self.canvas_wave.draw()

        self.ax_fft.clear()
        freqs, magnitude = analysis.compute_fft(data, sr)
        self.ax_fft.plot(freqs, magnitude, linewidth=0.8, color="tab:blue", label="Actuel")
        self.ax_fft.set_title("Spectre de fréquence (FFT)")
        self.ax_fft.set_xlabel("Fréquence (Hz)")
        self.ax_fft.set_ylabel("Amplitude")
        for freq, mag in analysis.find_dominant_frequencies(freqs, magnitude):
            self.ax_fft.annotate(
                f"{freq:.0f} Hz", xy=(freq, mag), xytext=(0, 8), textcoords="offset points", fontsize=8, ha="center"
            )

        if self.reference_data is not None:
            freqs_ref, magnitude_ref = analysis.compute_fft(self.reference_data, self.reference_sample_rate)
            self.ax_fft.plot(
                freqs_ref, magnitude_ref, linewidth=0.8, linestyle="--", alpha=0.7, color="tab:orange", label="Référence"
            )
            self.ax_fft.legend(fontsize=8)
            self._update_comparison(freqs_ref, magnitude_ref, freqs, magnitude)
        else:
            self._clear_comparison()
        self.fig_fft.tight_layout(pad=3)
        self.canvas_fft.draw()

        self.ax_spec.clear()
        f_spec, t_spec, sxx_db = analysis.compute_spectrogram(data, sr)
        self.ax_spec.pcolormesh(t_spec, f_spec, sxx_db, shading="auto", cmap="viridis")
        self.ax_spec.set_title("Spectrogramme")
        self.ax_spec.set_xlabel("Temps (s)")
        self.ax_spec.set_ylabel("Fréquence (Hz)")
        self.fig_spec.tight_layout(pad=3)
        self.canvas_spec.draw()

    def _update_comparison(self, freqs_ref, magnitude_ref, freqs_cur, magnitude_cur):
        self._clear_comparison()
        comparisons = analysis.compare_dominant_frequencies(freqs_ref, magnitude_ref, freqs_cur, magnitude_cur)
        for freq_ref, freq_cur, delta_hz, delta_pct in comparisons:
            self.comparison_tree.insert(
                "", tk.END, values=(f"{freq_ref:.0f}", f"{freq_cur:.0f}", f"{delta_hz:+.0f}", f"{delta_pct:+.1f}")
            )

    def _clear_comparison(self):
        for row in self.comparison_tree.get_children():
            self.comparison_tree.delete(row)


def main():
    SonoscopeApp().mainloop()


if __name__ == "__main__":
    main()
