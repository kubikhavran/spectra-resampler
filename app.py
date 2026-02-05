import io
import re
import zipfile
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


@dataclass
class Spectrum:
    filename: str
    x: np.ndarray
    y: np.ndarray


def _sniff_decimal_and_sep(text_sample: str) -> Tuple[str, str]:
    """
    Heuristika: detekuje desetinnou tečku vs čárku a delimiter (whitespace / ; / , / tab).
    V reálných exportech bývá nejčastější whitespace nebo tab, někdy ; (Excel).
    """
    # Odstraň komentáře a prázdné řádky
    lines = []
    for line in text_sample.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(("#", "//")):
            continue
        lines.append(s)
        if len(lines) >= 50:
            break
    sample = "\n".join(lines)

    # Decimal: pokud se vyskytuje pattern "digit,digit" víc než "digit.digit", ber čárku.
    comma_dec = len(re.findall(r"\d,\d", sample))
    dot_dec = len(re.findall(r"\d\.\d", sample))
    decimal = "," if comma_dec > dot_dec else "."

    # Separator: zkus detekovat podle počtu výskytů
    # Pořadí preferencí: ;, \t, whitespace, ,
    semi = sample.count(";")
    tab = sample.count("\t")
    comma = sample.count(",")
    # Pozor: pokud decimal je ",", tak čárka jako delimiter je extrémně nepravděpodobná.
    # Proto čárku jako delimiter preferujeme jen když decimal je ".".
    if semi > 0:
        sep = ";"
    elif tab > 0:
        sep = "\t"
    else:
        # whitespace jako default (pandas sep=r"\s+")
        sep = r"\s+"
        if decimal == "." and comma > 0 and semi == 0 and tab == 0:
            # může být CSV s čárkou jako delimiter, ale jen když decimal je "."
            sep = ","
    return decimal, sep


def read_spectrum_txt(uploaded_file) -> Spectrum:
    """
    Načte 2sloupcový txt (x, y), zvládne desetinnou čárku i tečku.
    Ignoruje prázdné řádky a komentáře začínající # nebo //.
    """
    raw = uploaded_file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        # fallback pro různé exporty
        text = raw.decode("latin-1", errors="replace")

    # Sniff jen z prvních pár KB
    sample = text[:8000]
    decimal, sep = _sniff_decimal_and_sep(sample)

    # pandas umí decimal=',' + sep=';' nebo sep=r'\s+'
    # comment umí jen jeden znak, tak odstraníme // ručně filtrem níže.
    df = pd.read_csv(
        io.StringIO(text),
        sep=sep,
        engine="python",
        header=None,
        decimal=decimal,
        comment="#",
        skip_blank_lines=True,
    )

    # Vyhoď řádky, které nejsou 2 numerické hodnoty (typicky hlavičky, jednotky)
    # Převeď na numeric a dropni NaN
    if df.shape[1] < 2:
        # zkus fallback: whitespace split bez pandas, když je to divné
        rows = []
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("//"):
                continue
            parts = re.split(r"[;\t, ]+", s)
            if len(parts) < 2:
                continue
            rows.append(parts[:2])
        df = pd.DataFrame(rows)

    df = df.iloc[:, :2].copy()
    df.columns = ["x", "y"]

    # Odstraň řádky se // komentářem
    # (pandas neumí comment='//' přímo)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"])

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    if x.size < 2:
        raise ValueError("Soubor neobsahuje dostatek numerických dat ve 2 sloupcích.")

    # Seřaď podle x a odstraň duplicitní x (pro interpolaci)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Odstraň duplicity x (ponechá první výskyt)
    uniq_mask = np.concatenate(([True], np.diff(x) != 0))
    x = x[uniq_mask]
    y = y[uniq_mask]

    return Spectrum(filename=uploaded_file.name, x=x, y=y)


def intersection_range(spectra: List[Spectrum]) -> Tuple[float, float]:
    """
    Najde průnik rozsahů X. Funguje i když mají některá spektra klesající osy (už jsou seřazená).
    """
    mins = [float(np.min(s.x)) for s in spectra]
    maxs = [float(np.max(s.x)) for s in spectra]
    x_min = max(mins)
    x_max = min(maxs)
    if not (x_max > x_min):
        raise ValueError(
            f"Neexistuje společný průnik osy X. (x_min={x_min}, x_max={x_max})"
        )
    return x_min, x_max


def resample_spectrum(s: Spectrum, x_new: np.ndarray) -> Spectrum:
    """
    Lineární interpolace na x_new.
    """
    y_new = np.interp(x_new, s.x, s.y)
    return Spectrum(filename=s.filename, x=x_new, y=y_new)


def spectrum_to_txt_bytes(s: Spectrum, decimal: str = ".", sep: str = "\t") -> bytes:
    """
    Export do txt: 2 sloupce, default tab delimiter, decimal tečka.
    """
    df = pd.DataFrame({"x": s.x, "y": s.y})
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, sep=sep, float_format="%.10g")
    out = buf.getvalue()
    if decimal == ",":
        # převeď desetinnou tečku na čárku (pozor: je to safe, protože sep je tab)
        out = out.replace(".", ",")
    return out.encode("utf-8")


def build_zip(spectra: List[Spectrum], decimal_out: str, sep_out: str) -> bytes:
    """
    Zabalí všechna spektra do ZIPu se zachováním názvů.
    """
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for s in spectra:
            zf.writestr(s.filename, spectrum_to_txt_bytes(s, decimal=decimal_out, sep=sep_out))
    mem.seek(0)
    return mem.read()


def main():
    st.set_page_config(page_title="Spectra Resampler (Raman/IR)", layout="centered")
    st.title("Spectra Resampler (Raman / IR)")
    st.write(
        "Nahraj více **.txt** spekter (2 sloupce: X=vlnočty / osa, Y=intenzita). "
        "Aplikace najde společný průnik osy X, všechny spektra **přeinterpoluje** na stejný počet bodů "
        "a vrátí je jako ZIP."
    )

    with st.expander("Nastavení", expanded=True):
        n_points = st.number_input(
            "Počet bodů po resamplingu",
            min_value=10,
            max_value=200000,
            value=1500,
            step=100,
        )

        st.caption("Výstupní formát")
        col1, col2 = st.columns(2)
        with col1:
            decimal_out = st.selectbox("Desetinný oddělovač", options=[".", ","], index=0)
        with col2:
            sep_out_label = st.selectbox("Oddělovač sloupců", options=["TAB", "Mezera", "Středník"], index=0)
            sep_out = "\t" if sep_out_label == "TAB" else (" " if sep_out_label == "Mezera" else ";")

    files = st.file_uploader(
        "Nahraj .txt soubory (můžeš označit více najednou)",
        type=["txt"],
        accept_multiple_files=True,
    )

    if not files:
        st.info("Nahraj prosím alespoň 2 soubory (ale klidně to funguje i pro 1).")
        return

    if st.button("Zpracovat a vytvořit ZIP", type="primary"):
        try:
            # načtení
            spectra = []
            for f in files:
                spectra.append(read_spectrum_txt(f))

            # průnik
            x_min, x_max = intersection_range(spectra)

            # nové x (rovnoměrná mřížka)
            x_new = np.linspace(x_min, x_max, int(n_points), dtype=float)

            # resampling
            resampled = [resample_spectrum(s, x_new) for s in spectra]

            # zip
            zip_bytes = build_zip(resampled, decimal_out=decimal_out, sep_out=sep_out)

            st.success(
                f"Hotovo. Společný rozsah X: **{x_min:.6g} až {x_max:.6g}**, počet bodů: **{int(n_points)}**."
            )

            st.download_button(
                label="Stáhnout ZIP se zpracovanými spektry",
                data=zip_bytes,
                file_name="resampled_spectra.zip",
                mime="application/zip",
            )

            # malý náhled pro kontrolu
            with st.expander("Náhled (prvních 5 řádků prvního spektra)", expanded=False):
                preview = pd.DataFrame({"x": resampled[0].x, "y": resampled[0].y}).head(5)
                st.dataframe(preview, use_container_width=True)

        except Exception as e:
            st.error(f"Chyba při zpracování: {e}")


if __name__ == "__main__":
    main()
