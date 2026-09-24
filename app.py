"""Sonoscope : enregistrer des produits, faire des tests sonores et les analyser (interface Tkinter)."""

import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from sonoscope import analysis, audio_io


def _parse_timestamp(horodatage: str) -> datetime:
    return datetime.strptime(horodatage, "%Y%m%d_%H%M%S")  # noqa: DTZ007 - horodatage local, tri chronologique seulement


def _format_timestamp(horodatage: str) -> str:
    try:
        return _parse_timestamp(horodatage).strftime("%d/%m/%Y %H:%M")
    except ValueError:
        return horodatage


class AudioAnalysisView(ttk.Frame):
    """Onglets Forme d'onde / Spectre FFT / Spectrogramme, réutilisés par plusieurs pages."""

    def __init__(self, parent):
        super().__init__(parent)
        notebook = ttk.Notebook(self)
        notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig_wave, self.ax_wave, self.canvas_wave = self._add_tab(notebook, "Forme d'onde")
        self.fig_fft, self.ax_fft, self.canvas_fft = self._add_tab(notebook, "Spectre FFT")
        self.fig_spec, self.ax_spec, self.canvas_spec = self._add_tab(notebook, "Spectrogramme")

    @staticmethod
    def _add_tab(notebook, title):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=title)
        figure = Figure(figsize=(10, 6), dpi=100)
        ax = figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(figure, master=tab)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        return figure, ax, canvas

    def plot(self, data, sample_rate, reference=None):
        """Affiche (data, sample_rate). `reference` optionnel : (data_ref, sr_ref) superposée sur la FFT.

        Renvoie la liste de comparaison (fréquence_ref, fréquence_actuelle, écart Hz, écart %),
        vide si aucune référence n'est fournie.
        """
        self.ax_wave.clear()
        t = np.linspace(0, len(data) / sample_rate, len(data))
        self.ax_wave.plot(t, data, linewidth=0.8)
        self.ax_wave.set_title("Forme d'onde")
        self.ax_wave.set_xlabel("Temps (s)")
        self.ax_wave.set_ylabel("Amplitude")
        self.fig_wave.tight_layout(pad=3)
        self.canvas_wave.draw()

        self.ax_fft.clear()
        freqs, magnitude = analysis.compute_fft(data, sample_rate)
        self.ax_fft.plot(freqs, magnitude, linewidth=0.8, color="tab:blue", label="Actuel")
        self.ax_fft.set_title("Spectre de fréquence (FFT)")
        self.ax_fft.set_xlabel("Fréquence (Hz)")
        self.ax_fft.set_ylabel("Amplitude")
        for freq, mag in analysis.find_dominant_frequencies(freqs, magnitude):
            self.ax_fft.annotate(
                f"{freq:.0f} Hz", xy=(freq, mag), xytext=(0, 8), textcoords="offset points", fontsize=8, ha="center"
            )

        comparisons = []
        if reference is not None:
            ref_data, ref_sr = reference
            freqs_ref, magnitude_ref = analysis.compute_fft(ref_data, ref_sr)
            self.ax_fft.plot(
                freqs_ref,
                magnitude_ref,
                linewidth=0.8,
                linestyle="--",
                alpha=0.7,
                color="tab:orange",
                label="Référence",
            )
            self.ax_fft.legend(fontsize=8)
            comparisons = analysis.compare_dominant_frequencies(freqs_ref, magnitude_ref, freqs, magnitude)
        self.fig_fft.tight_layout(pad=3)
        self.canvas_fft.draw()

        self.ax_spec.clear()
        f_spec, t_spec, sxx_db = analysis.compute_spectrogram(data, sample_rate)
        self.ax_spec.pcolormesh(t_spec, f_spec, sxx_db, shading="auto", cmap="viridis")
        self.ax_spec.set_title("Spectrogramme")
        self.ax_spec.set_xlabel("Temps (s)")
        self.ax_spec.set_ylabel("Fréquence (Hz)")
        self.fig_spec.tight_layout(pad=3)
        self.canvas_spec.draw()

        return comparisons


class HomePage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=40)
        ttk.Label(self, text="Sonoscope", font=("", 22, "bold")).pack(pady=(20, 5))
        ttk.Label(self, text="Enregistrer et analyser le son de vos produits", font=("", 11)).pack(pady=(0, 30))

        ttk.Button(
            self, text="Enregistrer un produit", width=32, command=lambda: app.show_page(NewProductPage)
        ).pack(pady=8)
        ttk.Button(self, text="Faire un test", width=32, command=lambda: app.show_page(TestPage)).pack(pady=8)
        ttk.Button(
            self, text="Produits enregistrés", width=32, command=lambda: app.show_page(ProductsPage)
        ).pack(pady=8)


class NewProductPage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=30)
        self.app = app
        self._last_nom = None

        header = ttk.Frame(self)
        header.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(header, text="< Accueil", command=lambda: app.show_page(HomePage)).pack(side=tk.LEFT)
        ttk.Label(header, text="Enregistrer un produit", font=("", 16, "bold")).pack(side=tk.LEFT, padx=15)

        form = ttk.Frame(self, padding=(0, 30, 0, 0))
        form.pack(anchor=tk.W)

        ttk.Label(form, text="Nom du produit :").grid(row=0, column=0, sticky=tk.W, pady=6)
        self.nom_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.nom_var, width=40).grid(row=0, column=1, pady=6, padx=(10, 0))

        ttk.Label(form, text="Numéro / référence :").grid(row=1, column=0, sticky=tk.W, pady=6)
        self.numero_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.numero_var, width=40).grid(row=1, column=1, pady=6, padx=(10, 0))

        ttk.Label(form, text="Spécificité :").grid(row=2, column=0, sticky=tk.NW, pady=6)
        self.specificite_text = tk.Text(form, width=40, height=6)
        self.specificite_text.grid(row=2, column=1, pady=6, padx=(10, 0))

        ttk.Button(self, text="Enregistrer le produit", command=self.on_save).pack(anchor=tk.W, pady=(20, 5))
        self.status_var = tk.StringVar()
        ttk.Label(self, textvariable=self.status_var).pack(anchor=tk.W)
        self.test_btn = ttk.Button(
            self, text="Faire un test avec ce produit", command=self.on_go_test, state=tk.DISABLED
        )
        self.test_btn.pack(anchor=tk.W, pady=(10, 0))

    def on_show(self):
        self.nom_var.set("")
        self.numero_var.set("")
        self.specificite_text.delete("1.0", tk.END)
        self.status_var.set("")
        self._last_nom = None
        self.test_btn.config(state=tk.DISABLED)

    def on_save(self):
        nom = self.nom_var.get().strip()
        if not nom:
            messagebox.showwarning("Produit", "Le nom du produit est obligatoire.")
            return
        specificite = self.specificite_text.get("1.0", tk.END).strip()
        audio_io.register_product(nom, self.numero_var.get().strip(), specificite)
        self.status_var.set(f"Produit « {nom} » enregistré.")
        self._last_nom = nom
        self.test_btn.config(state=tk.NORMAL)

    def on_go_test(self):
        test_page = self.app.pages[TestPage]
        test_page.product_var.set(self._last_nom)
        self.app.show_page(TestPage)


class TestPage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app
        self.data = None
        self.sample_rate = None
        self.reference_data = None
        self.reference_sample_rate = None

        header = ttk.Frame(self)
        header.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(header, text="< Accueil", command=lambda: app.show_page(HomePage)).pack(side=tk.LEFT)
        ttk.Label(header, text="Faire un test", font=("", 14, "bold")).pack(side=tk.LEFT, padx=15)

        product_row = ttk.Frame(self, padding=(0, 10, 0, 0))
        product_row.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(product_row, text="Produit :").pack(side=tk.LEFT)
        self.product_var = tk.StringVar()
        self.product_combo = ttk.Combobox(product_row, textvariable=self.product_var, state="readonly", width=30)
        self.product_combo.pack(side=tk.LEFT, padx=(5, 10))
        ttk.Button(
            product_row, text="+ Nouveau produit", command=lambda: app.show_page(NewProductPage)
        ).pack(side=tk.LEFT)

        controls = ttk.Frame(self, padding=(0, 10, 0, 5))
        controls.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(controls, text="Durée (s) :").pack(side=tk.LEFT)
        self.duration_var = tk.IntVar(value=5)
        ttk.Spinbox(controls, from_=1, to=30, textvariable=self.duration_var, width=5).pack(side=tk.LEFT, padx=(0, 10))
        self.record_btn = ttk.Button(controls, text="Enregistrer", command=self.on_record)
        self.record_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="Importer un fichier...", command=self.on_import).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="Écouter", command=self.on_play).pack(side=tk.LEFT, padx=5)

        save_row = ttk.Frame(self, padding=(0, 0, 0, 5))
        save_row.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(save_row, text="Cas :").pack(side=tk.LEFT)
        self.cas_var = tk.StringVar()
        ttk.Combobox(
            save_row, textvariable=self.cas_var, values=["", "Cas 1", "Cas 2", "Autre"], state="readonly", width=10
        ).pack(side=tk.LEFT, padx=(5, 15))
        ttk.Label(save_row, text="Remarques :").pack(side=tk.LEFT)
        self.remarques_var = tk.StringVar()
        ttk.Entry(save_row, textvariable=self.remarques_var, width=40).pack(side=tk.LEFT, padx=(5, 15))
        ttk.Button(save_row, text="Sauvegarder ce test", command=self.on_save).pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Sélectionnez un produit puis enregistrez ou importez un son.")
        ttk.Label(self, textvariable=self.status_var).pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        ref_frame = ttk.LabelFrame(self, text="Comparaison à une référence", padding=(10, 5))
        ref_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        ref_header = ttk.Frame(ref_frame)
        ref_header.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(ref_header, text="Charger une référence...", command=self.on_load_reference).pack(side=tk.LEFT)
        self.reference_label_var = tk.StringVar(value="Aucune référence chargée")
        ttk.Label(ref_header, textvariable=self.reference_label_var, padding=(10, 0)).pack(side=tk.LEFT)

        columns = ("ref", "cur", "delta_hz", "delta_pct")
        headings = ("Référence (Hz)", "Actuel (Hz)", "Écart (Hz)", "Écart (%)")
        self.comparison_tree = ttk.Treeview(ref_frame, columns=columns, show="headings", height=4)
        for col, label in zip(columns, headings, strict=True):
            self.comparison_tree.heading(col, text=label)
            self.comparison_tree.column(col, width=140, anchor=tk.CENTER)
        self.comparison_tree.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))

        self.analysis_view = AudioAnalysisView(self)
        self.analysis_view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def on_show(self):
        products = [p["nom"] for p in audio_io.list_registered_products()]
        self.product_combo["values"] = products
        if products and self.product_var.get() not in products:
            self.product_var.set(products[0])

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
        self._refresh_plot()

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
        self.load_as_reference(data, sr, f"Référence : {Path(path).name}")

    def load_as_reference(self, data, sr, label):
        """Utilisé aussi par la page Produits pour comparer un test historique."""
        self.reference_data, self.reference_sample_rate = data, sr
        self.reference_label_var.set(label)
        if self.data is not None:
            self._refresh_plot()

    def on_play(self):
        if self.data is None:
            return
        import sounddevice as sd

        sd.play(self.data, self.sample_rate)

    def on_save(self):
        if self.data is None:
            return
        if not self.product_var.get():
            messagebox.showwarning("Produit", "Sélectionnez ou créez d'abord un produit.")
            return
        freqs, magnitude = analysis.compute_fft(self.data, self.sample_rate)
        dominant = analysis.find_dominant_frequencies(freqs, magnitude)
        path = audio_io.save_recording(
            self.data,
            self.sample_rate,
            self.product_var.get(),
            cas=self.cas_var.get(),
            remarques=self.remarques_var.get(),
            dominant_frequencies=dominant,
        )
        messagebox.showinfo("Sauvegardé", f"Test sauvegardé :\n{path}")

    def _refresh_plot(self):
        reference = None
        if self.reference_data is not None:
            reference = (self.reference_data, self.reference_sample_rate)
        comparisons = self.analysis_view.plot(self.data, self.sample_rate, reference=reference)

        for row in self.comparison_tree.get_children():
            self.comparison_tree.delete(row)
        for freq_ref, freq_cur, delta_hz, delta_pct in comparisons:
            self.comparison_tree.insert(
                "", tk.END, values=(f"{freq_ref:.0f}", f"{freq_cur:.0f}", f"{delta_hz:+.0f}", f"{delta_pct:+.1f}")
            )


class ProductsPage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app
        self._products = []
        self._records = []

        header = ttk.Frame(self)
        header.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(header, text="< Accueil", command=lambda: app.show_page(HomePage)).pack(side=tk.LEFT)
        ttk.Label(header, text="Produits enregistrés", font=("", 14, "bold")).pack(side=tk.LEFT, padx=15)

        filter_row = ttk.Frame(self, padding=(0, 10))
        filter_row.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(filter_row, text="Filtrer (nom, numéro, spécificité) :").pack(side=tk.LEFT)
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add("write", lambda *_args: self._refresh_products_list())
        ttk.Entry(filter_row, textvariable=self.filter_var, width=30).pack(side=tk.LEFT, padx=(5, 0))

        body = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.Frame(body)
        columns = ("nom", "numero", "specificite", "nb_tests")
        headings = ("Nom", "Numéro", "Spécificité", "Tests")
        self.products_tree = ttk.Treeview(left, columns=columns, show="headings", height=20)
        for col, label in zip(columns, headings, strict=True):
            self.products_tree.heading(col, text=label)
            self.products_tree.column(col, width=180 if col == "specificite" else 110, anchor=tk.W)
        self.products_tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.products_tree.bind("<<TreeviewSelect>>", lambda _event: self._on_product_selected())
        body.add(left, weight=1)

        right = ttk.Frame(body)
        columns2 = ("date", "cas", "duree", "freq", "remarques")
        headings2 = ("Date", "Cas", "Durée (s)", "Fréq. dominante (Hz)", "Remarques")
        self.tests_tree = ttk.Treeview(right, columns=columns2, show="headings", height=5)
        for col, label in zip(columns2, headings2, strict=True):
            self.tests_tree.heading(col, text=label)
            self.tests_tree.column(col, width=130, anchor=tk.CENTER)
        self.tests_tree.pack(side=tk.TOP, fill=tk.X)

        actions = ttk.Frame(right, padding=(0, 5))
        actions.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(actions, text="Voir ce test", command=self._view_selected_test).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(
            actions, text="Utiliser comme référence (page Faire un test)", command=self._use_selected_as_reference
        ).pack(side=tk.LEFT)

        self.fig_trend = Figure(figsize=(8, 2.6), dpi=100)
        self.ax_trend = self.fig_trend.add_subplot(111)
        self.canvas_trend = FigureCanvasTkAgg(self.fig_trend, master=right)
        self.canvas_trend.get_tk_widget().pack(side=tk.TOP, fill=tk.X, pady=(5, 5))

        self.analysis_view = AudioAnalysisView(right)
        self.analysis_view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        body.add(right, weight=2)

    def on_show(self):
        self._products = audio_io.list_registered_products()
        self._refresh_products_list()

    def _refresh_products_list(self):
        query = self.filter_var.get().strip().lower()
        for row in self.products_tree.get_children():
            self.products_tree.delete(row)
        for product in self._products:
            haystack = " ".join(
                [product.get("nom", ""), product.get("numero", ""), product.get("specificite", "")]
            ).lower()
            if query and query not in haystack:
                continue
            self.products_tree.insert(
                "",
                tk.END,
                iid=product["folder"],
                values=(
                    product.get("nom", ""),
                    product.get("numero", ""),
                    product.get("specificite", ""),
                    product.get("nb_tests", 0),
                ),
            )

    def _on_product_selected(self):
        for row in self.tests_tree.get_children():
            self.tests_tree.delete(row)
        self._records = []

        selection = self.products_tree.selection()
        if selection:
            folder_name = selection[0]
            self._records = audio_io.list_recordings(folder_name)
            for record in self._records:
                peaks = record.get("frequences_dominantes") or []
                freq_text = f"{peaks[0][0]:.0f}" if peaks else "-"
                self.tests_tree.insert(
                    "",
                    tk.END,
                    values=(
                        _format_timestamp(record["horodatage"]),
                        record.get("cas", ""),
                        record.get("duree_sec", "-"),
                        freq_text,
                        record.get("remarques", ""),
                    ),
                )
        self._redraw_trend()

    def _redraw_trend(self):
        self.ax_trend.clear()
        self.ax_trend.set_title("Fréquence dominante dans le temps")
        self.ax_trend.set_xlabel("Date")
        self.ax_trend.set_ylabel("Fréquence (Hz)")

        points = [
            (_parse_timestamp(record["horodatage"]), record["frequences_dominantes"][0][0])
            for record in self._records
            if record.get("frequences_dominantes")
        ]
        if points:
            points.sort(key=lambda point: point[0])
            dates, freqs = zip(*points, strict=True)
            self.ax_trend.plot(dates, freqs, marker="o", linewidth=1)
            self.fig_trend.autofmt_xdate()
        self.fig_trend.tight_layout(pad=3)
        self.canvas_trend.draw()

    def _selected_test_record(self):
        selection = self.tests_tree.selection()
        if not selection:
            messagebox.showinfo("Tests", "Sélectionnez d'abord un test dans la liste.")
            return None
        index = self.tests_tree.index(selection[0])
        return self._records[index]

    def _view_selected_test(self):
        record = self._selected_test_record()
        if record is None:
            return
        data, sr = audio_io.load_audio_file(str(record["path"]))
        self.analysis_view.plot(data, sr)

    def _use_selected_as_reference(self):
        record = self._selected_test_record()
        if record is None:
            return
        data, sr = audio_io.load_audio_file(str(record["path"]))
        self.app.pages[TestPage].load_as_reference(data, sr, f"Référence : {record['path'].name}")
        self.app.show_page(TestPage)


class SonoscopeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sonoscope")
        self.geometry("1150x980")

        container = ttk.Frame(self)
        container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        self.pages = {}
        for page_class in (HomePage, NewProductPage, TestPage, ProductsPage):
            page = page_class(container, self)
            self.pages[page_class] = page
            page.grid(row=0, column=0, sticky="nsew")

        self.show_page(HomePage)

    def show_page(self, page_class):
        page = self.pages[page_class]
        if hasattr(page, "on_show"):
            page.on_show()
        page.tkraise()


def main():
    SonoscopeApp().mainloop()


if __name__ == "__main__":
    main()
