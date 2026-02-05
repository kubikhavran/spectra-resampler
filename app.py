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


def _clean_lines(text: str, max_lines: int = 200) -> List[str]:
    """Vrátí pár prvních relevantních řádků bez prázdných a komentářů."""
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("//"):
            continue
        lines.append(s)
        if len(lines) >= max_lines:
            break
    return lines


def _sniff_decimal_and_sep(text_sample: str) -> Tuple[str, str]:
    """
    Heuristika: detekuje desetinnou čárku/tečku a separator.
    """
    sample_lines = _clean_lines(text_sample, max_lines=50)
    sample = "\n".join(sample_lines)

    comma_dec = len(re.findall(r"\d,\d", sample))
    dot_dec = len(re.findall(r"\d\.\d", sample))
    decimal = "," if comma_dec > dot_dec else "."

    semi = sample.count(";")
    tab = sample.count("\t")
    comma = sample.count(",")

    if semi > 0:
        sep = ";"
    elif tab > 0:
        sep = "\t"
    else:
        sep = r"\s+"
        if decimal == "." and comma > 0 and semi == 0 and tab == 0:
            sep = ","

    return decimal, sep


def read_spectrum_txt(uploaded_file) -> Spectrum:
    """
    Načte 2sloupcový txt (x, y).
    """
    raw = uploaded_file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="replace")

    sample = text[:8000]
    decimal, sep = _sniff_decimal_and_sep(sample)

    df = None
    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=sep,
            engine="python",
            header=None,
            decimal=decimal,
            comment="#",
            skip_blank_lines=True,
        )
    except Exception:
        df = None

    if df is None or df.shape[1] < 2:
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

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"])

    if len(df) < 2:
        raise ValueError(f"{uploaded_file.name}: nenašel jsem dost numerických dat (2 sloupce).")

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    uniq_mask = np.concatenate(([True], np.diff(x) != 0))
    x = x[uniq_mask]
    y = y[uniq_mask]

    if x.size < 2:
        raise ValueError(f"{uploaded_file.name}: po vyčištění zůstalo méně než 2 body.")

    return Spectrum(filename=uploaded_file.name, x=x, y=y)


def resample_spectrum_keep_full_range(s: Spectrum, n_points: int) -> Spectrum:
    """
    Přeinterpoluje spektrum na n_points bodů v JEHO vlastním rozsahu.
    """
    x_min = float(np.min(s.x))
    x_max = float(np.max(s.x))
    if not (x_max > x_min):
        raise ValueError(f"{s.filename}: neplatný rozsah osy X.")

    x_new = np.linspace(x_min, x_max, int(n_points), dtype=float)
    y_new = np.interp(x_new, s.x, s.y)
    return Spectrum(filename=s.filename, x=x_new, y=y_new)


def normalize_spectrum_max(s: Spectrum) -> Spectrum:
    """
    Vydělí celé spektrum jeho maximální hodnotou Y, takže maximum bude 1.
    """
    max_y = np.max(s.y)
    # Ošetření dělení nulou, pokud by spektrum bylo prázdné nebo nulové
    if max_y != 0:
        new_y = s.y / max_y
    else:
        new_y = s.y
    
    return Spectrum(filename=s.filename, x=s.x, y=new_y)


def spectrum_to_txt_bytes(s: Spectrum, decimal: str = ".", sep: str = "\t") -> bytes:
    df = pd.DataFrame({"x": s.x, "y": s.y})
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, sep=sep, float_format="%.10g")
    out = buf.getvalue()

    if decimal == ",":
        out = out.replace(".", ",")

    return out.encode("utf-8")


def build_zip(spectra: List[Spectrum], decimal_out: str, sep_out: str) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for s in spectra:
            zf.writestr(s.filename, spectrum_to_txt_bytes(s, decimal=decimal_out, sep=sep_out))
    mem.seek(0)
    return mem.read()


def main():
    st.set_page_config(page_title="Spectra Resampler & Normalizer", layout="centered")
    st.title("Spectra Resampler & Normalizer")
    st.write(
        "Nahraj více **.txt** spekter. Aplikace je přeinterpoluje na zvolený počet bodů "
        "a volitelně **normalizuje** (nejvyšší pík = 1)."
    )

    with st.expander("Nastavení", expanded=True):
        col_set1, col_set2 = st.columns(2)
        with col_set1:
            n_points = st.number_input(
                "Počet bodů (resampling)",
                min_value=10,
                max_value=200000,
                value=1500,
                step=100,
            )
        with col_set2:
            do_normalize = st.checkbox(
                "Normalizovat intenzitu (max = 1)",
                value=True,
                help="Vydělí y hodnoty nejvyšší hodnotou ve spektru."
            )

        st.caption("Výstupní formát TXT")
        col1, col2 = st.columns(2)
        with col1:
            decimal_out = st.selectbox("Desetinný oddělovač", options=[".", ","], index=0)
        with col2:
            sep_out_label = st.selectbox("Oddělovač sloupců", options=["TAB", "Mezera", "Středník"], index=0)
            sep_out = "\t" if sep_out_label == "TAB" else (" " if sep_out_label == "Mezera" else ";")

    files = st.file_uploader(
        "Nahraj .txt soubory",
        type=["txt"],
        accept_multiple_files=True,
    )

    if not files:
        st.info("Nahraj prosím alespoň 1 soubor.")
        return

    if st.button("Zpracovat a vytvořit ZIP", type="primary"):
        try:
            spectra = [read_spectrum_txt(f) for f in files]

            # 1. Resampling
            processed = [resample_spectrum_keep_full_range(s, int(n_points)) for s in spectra]

            # 2. Normalizace (pokud je zaškrtnuto)
            if do_normalize:
                processed = [normalize_spectrum_max(s) for s in processed]

            zip_bytes = build_zip(processed, decimal_out=decimal_out, sep_out=sep_out)

            norm_msg = " a **normalizováno** (max=1)" if do_normalize else ""
            st.success(f"Hotovo. Převedeno na {int(n_points)} bodů{norm_msg}.")

            st.download_button(
                label="Stáhnout ZIP",
                data=zip_bytes,
                file_name="processed_spectra.zip",
                mime="application/zip",
            )

            # Náhled
            with st.expander(f"Náhled: {processed[0].filename}", expanded=True):
                preview = pd.DataFrame({"x": processed[0].x, "y": processed[0].y})
                
                # Zobrazíme graf pro rychlou kontrolu normalizace
                st.line_chart(preview, x="x", y="y")
                
                st.write("**Data (prvních 5 řádků):**")
                st.dataframe(preview.head(5), use_container_width=True)

        except Exception as e:
            st.error(f"Chyba při zpracování: {e}")


if __name__ == "__main__":
    main()
