"""Sonoscope : enregistrer des produits, faire des tests sonores et les analyser (interface Tkinter)."""

import re
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

        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        self.zoom_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            toolbar,
            text="Zoom auto sur les pics (Spectre FFT)",
            variable=self.zoom_var,
            command=self._apply_fft_zoom,
        ).pack(side=tk.LEFT, padx=5, pady=3)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig_wave, self.ax_wave, self.canvas_wave = self._add_tab(self.notebook, "Forme d'onde")
        self.fig_fft, self.ax_fft, self.canvas_fft = self._add_tab(self.notebook, "Spectre FFT")
        self.fig_spec, self.ax_spec, self.canvas_spec = self._add_tab(self.notebook, "Spectrogramme")

        self._fft_zoom_xlim = None
        self._fft_full_xlim = None

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

        Renvoie la liste de comparaison (fréquence_ref, fréquence_actuelle, écart Hz, écart %, fiable),
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
        current_peaks = analysis.find_dominant_frequencies(freqs, magnitude)
        self.ax_fft.plot(freqs, magnitude, linewidth=1.1, color="tab:blue", label="Actuel")
        if current_peaks:
            self.ax_fft.scatter(
                [f for f, _ in current_peaks], [m for _, m in current_peaks], color="tab:blue", zorder=5, s=35
            )
        self.ax_fft.set_title("Spectre de fréquence (FFT)")
        self.ax_fft.set_xlabel("Fréquence (Hz)")
        self.ax_fft.set_ylabel("Amplitude")
        self.ax_fft.grid(True, alpha=0.3)

        max_freq_of_interest = max((f for f, _ in current_peaks), default=0.0)

        comparisons = []
        if reference is not None:
            ref_data, ref_sr = reference
            freqs_ref, magnitude_ref = analysis.compute_fft(ref_data, ref_sr)
            reference_peaks = analysis.find_dominant_frequencies(freqs_ref, magnitude_ref)
            self.ax_fft.plot(
                freqs_ref,
                magnitude_ref,
                linewidth=1.1,
                linestyle="--",
                alpha=0.8,
                color="tab:orange",
                label="Référence",
            )
            if reference_peaks:
                self.ax_fft.scatter(
                    [f for f, _ in reference_peaks],
                    [m for _, m in reference_peaks],
                    color="tab:orange",
                    marker="D",
                    zorder=5,
                    s=35,
                )
                max_freq_of_interest = max(max_freq_of_interest, max(f for f, _ in reference_peaks))
            self.ax_fft.legend(fontsize=9, loc="upper right")
            comparisons = analysis.compare_dominant_frequencies(freqs_ref, magnitude_ref, freqs, magnitude)

        # Zoom automatique : tout le contenu utile (moteurs/roulements) est en général très en
        # dessous du maximum théorique (22 kHz à 44.1 kHz), sans ce zoom les pics sont écrasés
        # dans un coin du graphe. La case à cocher permet de repasser en vue complète.
        self._fft_full_xlim = (0, freqs[-1] if len(freqs) else 1.0)
        self._fft_zoom_xlim = (0, max_freq_of_interest * 1.4) if max_freq_of_interest > 0 else self._fft_full_xlim

        self.fig_fft.tight_layout(pad=3)
        self._apply_fft_zoom()

        self.ax_spec.clear()
        f_spec, t_spec, sxx_db = analysis.compute_spectrogram(data, sample_rate)
        self.ax_spec.pcolormesh(t_spec, f_spec, sxx_db, shading="auto", cmap="viridis")
        self.ax_spec.set_title("Spectrogramme")
        self.ax_spec.set_xlabel("Temps (s)")
        self.ax_spec.set_ylabel("Fréquence (Hz)")
        self.fig_spec.tight_layout(pad=3)
        self.canvas_spec.draw()

        return comparisons

    def _apply_fft_zoom(self):
        if self._fft_zoom_xlim is None:
            return
        xlim = self._fft_zoom_xlim if self.zoom_var.get() else self._fft_full_xlim
        self.ax_fft.set_xlim(*xlim)
        self.canvas_fft.draw()


class HomePage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=40)
        ttk.Label(self, text="Sonoscope", font=("", 22, "bold")).pack(pady=(20, 5))
        ttk.Label(self, text="Enregistrer et analyser le son de vos produits", font=("", 11)).pack(pady=(0, 30))

        ttk.Button(self, text="Enregistrer un produit", width=32, command=app.open_new_product).pack(pady=8)
        ttk.Button(self, text="Faire un test", width=32, command=lambda: app.show_page(TestPage)).pack(pady=8)
        ttk.Button(
            self, text="Produits enregistrés", width=32, command=lambda: app.show_page(ProductsPage)
        ).pack(pady=8)


class NewProductPage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=30)
        self.app = app
        self._last_folder = None
        self.editing_folder = None

        header = ttk.Frame(self)
        header.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(header, text="< Accueil", command=lambda: app.show_page(HomePage)).pack(side=tk.LEFT)
        self.title_var = tk.StringVar(value="Enregistrer un produit")
        ttk.Label(header, textvariable=self.title_var, font=("", 16, "bold")).pack(side=tk.LEFT, padx=15)

        form = ttk.Frame(self, padding=(0, 30, 0, 0))
        form.pack(anchor=tk.W)

        ttk.Label(form, text="Nom du produit :").grid(row=0, column=0, sticky=tk.W, pady=6)
        self.nom_var = tk.StringVar()
        self.nom_entry = ttk.Entry(form, textvariable=self.nom_var, width=40)
        self.nom_entry.grid(row=0, column=1, pady=6, padx=(10, 0), sticky=tk.W)

        ttk.Label(form, text="Type :").grid(row=1, column=0, sticky=tk.W, pady=6)
        self.type_var = tk.StringVar()
        ttk.Combobox(form, textvariable=self.type_var, values=["KT", "LT"], state="readonly", width=10).grid(
            row=1, column=1, pady=6, padx=(10, 0), sticky=tk.W
        )

        ttk.Label(form, text="Numéro (4 chiffres) :").grid(row=2, column=0, sticky=tk.W, pady=6)
        self.numero_var = tk.StringVar()
        self.numero_entry = ttk.Entry(form, textvariable=self.numero_var, width=10)
        self.numero_entry.grid(row=2, column=1, pady=6, padx=(10, 0), sticky=tk.W)

        ttk.Label(form, text="Date (JJ/MM/AAAA) :").grid(row=3, column=0, sticky=tk.W, pady=6)
        self.date_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.date_var, width=12).grid(row=3, column=1, pady=6, padx=(10, 0), sticky=tk.W)

        ttk.Label(form, text="Spécificité :").grid(row=4, column=0, sticky=tk.NW, pady=6)
        self.specificite_text = tk.Text(form, width=40, height=6)
        self.specificite_text.grid(row=4, column=1, pady=6, padx=(10, 0))

        self.save_btn = ttk.Button(self, text="Enregistrer le produit", command=self.on_save)
        self.save_btn.pack(anchor=tk.W, pady=(20, 5))
        self.status_var = tk.StringVar()
        ttk.Label(self, textvariable=self.status_var).pack(anchor=tk.W)
        self.test_btn = ttk.Button(
            self, text="Faire un test avec ce produit", command=self.on_go_test, state=tk.DISABLED
        )
        self.test_btn.pack(anchor=tk.W, pady=(10, 0))

    def on_show(self):
        if self.editing_folder:
            self._load_for_edit(self.editing_folder)
        else:
            self._reset_form()

    def start_create(self):
        self.editing_folder = None

    def start_edit(self, folder_name):
        self.editing_folder = folder_name

    def _today(self):
        return datetime.now().strftime("%d/%m/%Y")  # noqa: DTZ005 - date locale, champ pré-rempli mais éditable

    def _reset_form(self):
        self.nom_entry.config(state=tk.NORMAL)
        self.nom_var.set("")
        self.type_var.set("")
        self.numero_entry.config(state=tk.NORMAL)
        self.numero_var.set("")
        self.date_var.set(self._today())
        self.specificite_text.delete("1.0", tk.END)
        self.status_var.set("")
        self._last_folder = None
        self.test_btn.config(state=tk.DISABLED)
        self.title_var.set("Enregistrer un produit")
        self.save_btn.config(text="Enregistrer le produit")

    def _load_for_edit(self, folder_name):
        product = audio_io.get_product(folder_name) or {"nom": folder_name}
        self.nom_var.set(product.get("nom", folder_name))
        self.nom_entry.config(state="disabled")
        self.type_var.set(product.get("type_test", ""))
        self.numero_var.set(product.get("numero", ""))
        self.numero_entry.config(state="disabled")
        self.date_var.set(product.get("date_produit") or self._today())
        self.specificite_text.delete("1.0", tk.END)
        self.specificite_text.insert("1.0", product.get("specificite", ""))
        self.status_var.set("")
        self._last_folder = folder_name
        self.test_btn.config(state=tk.NORMAL)
        self.title_var.set("Modifier le produit")
        self.save_btn.config(text="Enregistrer les modifications")

    def on_save(self):
        nom = self.nom_var.get().strip()
        if not nom:
            messagebox.showwarning("Produit", "Le nom du produit est obligatoire.")
            return
        numero = self.numero_var.get().strip()
        if not re.fullmatch(r"\d{4}", numero):
            messagebox.showwarning("Produit", "Le numéro est obligatoire et doit comporter exactement 4 chiffres.")
            return
        type_test = self.type_var.get()
        if not type_test:
            messagebox.showwarning("Produit", "Sélectionnez un type (KT ou LT).")
            return
        date_produit = self.date_var.get().strip()
        if date_produit:
            try:
                datetime.strptime(date_produit, "%d/%m/%Y")  # noqa: DTZ007 - validation de format seulement
            except ValueError:
                messagebox.showwarning("Produit", "Date invalide, format attendu : JJ/MM/AAAA.")
                return
        specificite = self.specificite_text.get("1.0", tk.END).strip()

        if self.editing_folder:
            audio_io.update_product(self.editing_folder, numero, specificite, type_test, date_produit)
            self._last_folder = self.editing_folder
            self.status_var.set(f"Produit « {nom} » mis à jour.")
        else:
            if audio_io.product_exists(nom, numero):
                messagebox.showwarning(
                    "Produit",
                    f"Un produit « {nom} » avec le numéro {numero} existe déjà.\n"
                    "Un même nom peut être réutilisé, mais pas avec le même numéro.",
                )
                return
            folder = audio_io.register_product(nom, numero, specificite, type_test, date_produit)
            self._last_folder = folder.name
            self.status_var.set(f"Produit « {nom} » enregistré.")

        self.test_btn.config(state=tk.NORMAL)

    def on_go_test(self):
        self.app.pages[TestPage].select_product_by_folder(self._last_folder)
        self.app.show_page(TestPage)


class TestPage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app
        self.data = None
        self.sample_rate = None
        self.reference_data = None
        self.reference_sample_rate = None
        self._product_by_label = {}

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
        ttk.Button(product_row, text="+ Nouveau produit", command=app.open_new_product).pack(side=tk.LEFT)

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

        columns = ("ref", "cur", "delta_hz", "delta_pct", "statut")
        headings = ("Référence (Hz)", "Actuel (Hz)", "Écart (Hz)", "Écart (%)", "Statut")
        self.comparison_tree = ttk.Treeview(ref_frame, columns=columns, show="headings", height=4)
        for col, label in zip(columns, headings, strict=True):
            self.comparison_tree.heading(col, text=label)
            self.comparison_tree.column(col, width=140 if col != "statut" else 170, anchor=tk.CENTER)
        self.comparison_tree.tag_configure("incertain", foreground="#999999")
        self.comparison_tree.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))

        self.analysis_view = AudioAnalysisView(self)
        self.analysis_view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def on_show(self):
        self._product_by_label = {}
        labels = []
        for product in audio_io.list_registered_products():
            nom = product.get("nom", product["folder"])
            numero = product.get("numero", "")
            label = f"{nom} ({numero})" if numero else f"{nom} [{product['folder']}]"
            self._product_by_label[label] = product["folder"]
            labels.append(label)
        self.product_combo["values"] = labels
        if labels and self.product_var.get() not in labels:
            self.product_var.set(labels[0])

    def select_product_by_folder(self, folder_name):
        """Utilisé par la page Nouveau produit après création/édition."""
        self.on_show()
        for label, folder in self._product_by_label.items():
            if folder == folder_name:
                self.product_var.set(label)
                return

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
        folder_name = self._product_by_label.get(self.product_var.get())
        if not folder_name:
            messagebox.showwarning("Produit", "Sélectionnez ou créez d'abord un produit.")
            return
        freqs, magnitude = analysis.compute_fft(self.data, self.sample_rate)
        dominant = analysis.find_dominant_frequencies(freqs, magnitude)
        path = audio_io.save_recording(
            self.data,
            self.sample_rate,
            folder_name,
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
        for freq_ref, freq_cur, delta_hz, delta_pct, fiable in comparisons:
            statut = "Dérive" if fiable else "Pic différent (?)"
            self.comparison_tree.insert(
                "",
                tk.END,
                values=(f"{freq_ref:.0f}", f"{freq_cur:.0f}", f"{delta_hz:+.0f}", f"{delta_pct:+.1f}", statut),
                tags=() if fiable else ("incertain",),
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
        columns = ("nom", "type", "numero", "date", "specificite", "nb_tests")
        headings = ("Nom", "Type", "Numéro", "Date", "Spécificité", "Tests")
        widths = {"nom": 130, "type": 50, "numero": 70, "date": 90, "specificite": 160, "nb_tests": 50}
        self.products_tree = ttk.Treeview(left, columns=columns, show="headings", height=18)
        for col, label in zip(columns, headings, strict=True):
            self.products_tree.heading(col, text=label)
            self.products_tree.column(col, width=widths[col], anchor=tk.W)
        self.products_tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.products_tree.bind("<<TreeviewSelect>>", lambda _event: self._on_product_selected())

        ttk.Button(left, text="Modifier ce produit", command=self._edit_selected_product).pack(
            side=tk.TOP, fill=tk.X, pady=(5, 0)
        )
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

        self.analysis_view = AudioAnalysisView(right)
        self.analysis_view.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(5, 0))

        trend_tab = ttk.Frame(self.analysis_view.notebook)
        self.analysis_view.notebook.insert(0, trend_tab, text="Tendance")
        self.fig_trend = Figure(figsize=(10, 6), dpi=100)
        self.ax_trend = self.fig_trend.add_subplot(111)
        self.canvas_trend = FigureCanvasTkAgg(self.fig_trend, master=trend_tab)
        self.canvas_trend.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.analysis_view.notebook.select(trend_tab)

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
                    product.get("type_test", ""),
                    product.get("numero", ""),
                    product.get("date_produit", ""),
                    product.get("specificite", ""),
                    product.get("nb_tests", 0),
                ),
            )

    def _edit_selected_product(self):
        selection = self.products_tree.selection()
        if not selection:
            messagebox.showinfo("Produits", "Sélectionnez d'abord un produit dans la liste.")
            return
        self.app.pages[NewProductPage].start_edit(selection[0])
        self.app.show_page(NewProductPage)

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

    def open_new_product(self):
        self.pages[NewProductPage].start_create()
        self.show_page(NewProductPage)


def main():
    SonoscopeApp().mainloop()


if __name__ == "__main__":
    main()
