"""Sonoscope : enregistrer un son (produit) et le décomposer en fréquences (interface Tkinter)."""

import threading
import tkinter as tk
from datetime import datetime
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
        self.geometry("1150x980")

        self.data = None
        self.sample_rate = None
        self.reference_data = None
        self.reference_sample_rate = None
        self.reference_label_var = tk.StringVar(value="Aucune référence chargée")

        self._build_controls()
        self._build_comparison()
        self._build_plots()

    def _build_controls(self):
        row1 = ttk.Frame(self, padding=(10, 10, 10, 5))
        row1.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(row1, text="Durée (s) :").pack(side=tk.LEFT)
        self.duration_var = tk.IntVar(value=5)
        ttk.Spinbox(row1, from_=1, to=30, textvariable=self.duration_var, width=5).pack(side=tk.LEFT, padx=(0, 10))

        self.record_btn = ttk.Button(row1, text="Enregistrer", command=self.on_record)
        self.record_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(row1, text="Importer un fichier...", command=self.on_import).pack(side=tk.LEFT, padx=5)
        ttk.Button(row1, text="Écouter", command=self.on_play).pack(side=tk.LEFT, padx=5)

        row2 = ttk.Frame(self, padding=(10, 0, 10, 5))
        row2.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(row2, text="Nom du produit :").pack(side=tk.LEFT)
        self.product_var = tk.StringVar()
        ttk.Entry(row2, textvariable=self.product_var, width=20).pack(side=tk.LEFT, padx=(5, 15))

        ttk.Label(row2, text="Cas :").pack(side=tk.LEFT)
        self.cas_var = tk.StringVar()
        ttk.Combobox(
            row2, textvariable=self.cas_var, values=["", "Cas 1", "Cas 2", "Autre"], state="readonly", width=10
        ).pack(side=tk.LEFT, padx=(5, 15))

        ttk.Label(row2, text="Remarques :").pack(side=tk.LEFT)
        self.remarques_var = tk.StringVar()
        ttk.Entry(row2, textvariable=self.remarques_var, width=30).pack(side=tk.LEFT, padx=(5, 15))

        ttk.Button(row2, text="Sauvegarder", command=self.on_save).pack(side=tk.LEFT)

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
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig_wave, self.ax_wave, self.canvas_wave = self._add_plot_tab("Forme d'onde")
        self.fig_fft, self.ax_fft, self.canvas_fft = self._add_plot_tab("Spectre FFT")
        self.fig_spec, self.ax_spec, self.canvas_spec = self._add_plot_tab("Spectrogramme")
        self._build_history_tab()

    def _add_plot_tab(self, title):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=title)

        figure = Figure(figsize=(10, 6), dpi=100)
        ax = figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(figure, master=tab)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        return figure, ax, canvas

    def _build_history_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Historique")

        top = ttk.Frame(tab, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="Produit :").pack(side=tk.LEFT)
        self.history_product_var = tk.StringVar()
        self.history_product_combo = ttk.Combobox(
            top, textvariable=self.history_product_var, state="readonly", width=25
        )
        self.history_product_combo.pack(side=tk.LEFT, padx=(5, 10))
        self.history_product_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_history())
        ttk.Button(top, text="Rafraîchir la liste des produits", command=self._refresh_products).pack(side=tk.LEFT)

        columns = ("date", "cas", "duree", "freq", "remarques")
        headings = ("Date", "Cas", "Durée (s)", "Fréq. dominante (Hz)", "Remarques")
        self.history_tree = ttk.Treeview(tab, columns=columns, show="headings", height=8)
        for col, label in zip(columns, headings, strict=True):
            self.history_tree.heading(col, text=label)
            self.history_tree.column(col, width=160, anchor=tk.CENTER)
        self.history_tree.pack(side=tk.TOP, fill=tk.X, padx=10)

        actions = ttk.Frame(tab, padding=10)
        actions.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(actions, text="Charger comme actuel", command=self._load_history_as_current).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        ttk.Button(actions, text="Charger comme référence", command=self._load_history_as_reference).pack(
            side=tk.LEFT
        )

        self.fig_history = Figure(figsize=(10, 3.5), dpi=100)
        self.ax_history = self.fig_history.add_subplot(111)
        self.canvas_history = FigureCanvasTkAgg(self.fig_history, master=tab)
        self.canvas_history.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        self._history_records = []
        self._refresh_products()

    def _refresh_products(self):
        products = audio_io.list_products()
        self.history_product_combo["values"] = products
        if products and self.history_product_var.get() not in products:
            self.history_product_var.set(products[0])
        elif not products:
            self.history_product_var.set("")
        self._refresh_history()

    def _refresh_history(self):
        for row in self.history_tree.get_children():
            self.history_tree.delete(row)

        product = self.history_product_var.get()
        self._history_records = audio_io.list_recordings(product) if product else []

        for record in self._history_records:
            peaks = record.get("frequences_dominantes") or []
            freq_text = f"{peaks[0][0]:.0f}" if peaks else "-"
            self.history_tree.insert(
                "",
                tk.END,
                values=(
                    self._format_timestamp(record["horodatage"]),
                    record.get("cas", ""),
                    record.get("duree_sec", "-"),
                    freq_text,
                    record.get("remarques", ""),
                ),
            )
        self._redraw_history_trend()

    def _redraw_history_trend(self):
        self.ax_history.clear()
        self.ax_history.set_title("Fréquence dominante dans le temps")
        self.ax_history.set_xlabel("Date")
        self.ax_history.set_ylabel("Fréquence (Hz)")

        points = [
            (self._parse_timestamp(record["horodatage"]), record["frequences_dominantes"][0][0])
            for record in self._history_records
            if record.get("frequences_dominantes")
        ]
        if points:
            points.sort(key=lambda point: point[0])
            dates, freqs = zip(*points, strict=True)
            self.ax_history.plot(dates, freqs, marker="o", linewidth=1)
            self.fig_history.autofmt_xdate()
        self.fig_history.tight_layout(pad=3)
        self.canvas_history.draw()

    @staticmethod
    def _parse_timestamp(horodatage: str) -> datetime:
        return datetime.strptime(horodatage, "%Y%m%d_%H%M%S")  # noqa: DTZ007 - horodatage local, tri chronologique seulement

    @classmethod
    def _format_timestamp(cls, horodatage: str) -> str:
        try:
            return cls._parse_timestamp(horodatage).strftime("%d/%m/%Y %H:%M")
        except ValueError:
            return horodatage

    def _selected_history_record(self):
        selection = self.history_tree.selection()
        if not selection:
            messagebox.showinfo("Historique", "Sélectionnez d'abord une ligne dans l'historique.")
            return None
        index = self.history_tree.index(selection[0])
        return self._history_records[index]

    def _load_history_as_current(self):
        record = self._selected_history_record()
        if record is None:
            return
        data, sr = audio_io.load_audio_file(str(record["path"]))
        self._on_audio_ready(data, sr, f"Historique chargé : {record['path'].name}")

    def _load_history_as_reference(self):
        record = self._selected_history_record()
        if record is None:
            return
        data, sr = audio_io.load_audio_file(str(record["path"]))
        self.reference_data, self.reference_sample_rate = data, sr
        self.reference_label_var.set(f"Référence : {record['path'].name}")
        if self.data is not None:
            self._plot()

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
        freqs, magnitude = analysis.compute_fft(self.data, self.sample_rate)
        dominant = analysis.find_dominant_frequencies(freqs, magnitude)
        path = audio_io.save_recording(
            self.data,
            self.sample_rate,
            self.product_var.get() or "produit",
            cas=self.cas_var.get(),
            remarques=self.remarques_var.get(),
            dominant_frequencies=dominant,
        )
        messagebox.showinfo("Sauvegardé", f"Enregistrement sauvegardé :\n{path}")
        self._refresh_products()

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
