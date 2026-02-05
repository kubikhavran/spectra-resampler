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
    Typicky: whitespace nebo tab, někdy ; (Excel). CSV s ',' delimiterem jen když decimal='.'.
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
        # default: whitespace
        sep = r"\s+"
        # CSV s ',' jako oddělovač jen když decimal='.'
        if decimal == "." and comma > 0 and semi == 0 and tab == 0:
            sep = ","

    return decimal, sep


def read_spectrum_txt(uploaded_file) -> Spectrum:
    """
    Načte 2sloupcový txt (x, y). Zvládne desetinnou čárku i tečku.
    Ignoruje prázdné řádky, komentáře (#, //) a hlavičky (nenumerické řádky).
    """
    raw = uploaded_file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="replace")

    sample = text[:8000]
    decimal, sep = _sniff_decimal_and_sep(sample)

    # Pokus 1: pandas read_csv
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

    # Fallback: ruční split (kdyby to mělo divné hlavičky / mix delimitérů)
    if df is None or df.shape[1] < 2:
        rows = []
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("//"):
                continue
            # split na ; , tab nebo whitespace
            parts = re.split(r"[;\t, ]+", s)
            if len(parts) < 2:
                continue
            rows.append(parts[:2])
        df = pd.DataFrame(rows)

    # Vezmi první 2 sloupce
    df = df.iloc[:, :2].copy()
    df.columns = ["x", "y"]

    # Odstraň řádky s // komentářem (pandas neumí comment='//')
    # + konverze na čísla
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"])

    if len(df) < 2:
        raise ValueError(f"{uploaded_file.name}: nenašel jsem dost numerických dat (2 sloupce).")

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    # Seřadit podle x (někdy bývá osa obráceně)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Odstranit duplicitní x (interpolace vyžaduje striktně rostoucí x)
    uniq_mask = np.concatenate(([True], np.diff(x) != 0))
    x = x[uniq_mask]
    y = y[uniq_mask]

    if x.size < 2:
        raise ValueError(f"{uploaded_file.name}: po vyčištění zůstalo méně než 2 body.")

    return Spectrum(filename=uploaded_file.name, x=x, y=y)


def resample_spectrum_keep_full_range(s: Spectrum, n_points: int) -> Spectrum:
    """
    Přeinterpoluje spektrum na n_points bodů v JEHO vlastním rozsahu (min(x) .. max(x)).
    Nic neořezává.
    """
    x_min = float(np.min(s.x))
    x_max = float(np.max(s.x))
    if not (x_max > x_min):
        raise ValueError(f"{s.filename}: neplatný rozsah osy X (x_min={x_min}, x_max={x_max}).")

    x_new = np.linspace(x_min, x_max, int(n_points), dtype=float)
    y_new = np.interp(x_new, s.x, s.y)
    return Spectrum(filename=s.filename, x=x_new, y=y_new)


def spectrum_to_txt_bytes(s: Spectrum, decimal: str = ".", sep: str = "\t") -> bytes:
    """
    Export do txt: 2 sloupce, bez hlavičky, default TAB.
    """
    df = pd.DataFrame({"x": s.x, "y": s.y})
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, sep=sep, float_format="%.10g")
    out = buf.getvalue()

    # Pokud chceš desetinnou čárku, bezpečně nahradíme tečky (sep je tab/mezera/středník)
    if decimal == ",":
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
        "Nahraj více **.txt** spekter (2 sloupce: X=vlnočty/osa, Y=intenzita). "
        "Aplikace **neořezává okraje** — každé spektrum resampluje v jeho původním rozsahu "
        "na stejný počet bodů a vrátí ZIP."
    )

    with st.expander("Nastavení", expanded=True):
        n_points = st.number_input(
            "Počet bodů po resamplingu (stejný pro všechny soubory)",
            min_value=10,
            max_value=200000,
            value=1500,
            step=100,
        )

        st.caption("Výstupní formát TXT")
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
        st.info("Nahraj prosím alespoň 1 soubor (klidně více).")
        return

    if st.button("Zpracovat a vytvořit ZIP", type="primary"):
        try:
            spectra = [read_spectrum_txt(f) for f in files]

            # Resampling bez ořezu: každé spektrum si zachová svůj min..max
            resampled = [resample_spectrum_keep_full_range(s, int(n_points)) for s in spectra]

            zip_bytes = build_zip(resampled, decimal_out=decimal_out, sep_out=sep_out)

            # Shrnutí rozsahů (pro kontrolu)
            ranges = [(s.filename, float(np.min(s.x)), float(np.max(s.x))) for s in resampled]
            st.success(f"Hotovo. Každý soubor má nyní **{int(n_points)}** bodů a zachovaný původní rozsah osy X.")

            st.download_button(
                label="Stáhnout ZIP se zpracovanými spektry",
                data=zip_bytes,
                file_name="resampled_spectra.zip",
                mime="application/zip",
            )

            with st.expander("Kontrola: rozsahy X po resamplingu", expanded=False):
                df_ranges = pd.DataFrame(ranges, columns=["soubor", "x_min", "x_max"])
                st.dataframe(df_ranges, use_container_width=True)

            with st.expander("Náhled: prvních 5 řádků prvního resamplovaného spektra", expanded=False):
                preview = pd.DataFrame({"x": resampled[0].x, "y": resampled[0].y}).head(5)
                st.dataframe(preview, use_container_width=True)

        except Exception as e:
            st.error(f"Chyba při zpracování: {e}")


if __name__ == "__main__":
    main()
