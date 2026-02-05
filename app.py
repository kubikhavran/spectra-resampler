import io
import re
import zipfile
from dataclasses import dataclass
from typing import List, Tuple, Optional

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
    """Heuristika: detekuje desetinnou čárku/tečku a separator."""
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
    """Načte 2sloupcový txt (x, y)."""
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

    # Seřadit podle x (někdy bývá osa obráceně, sjednotíme na rostoucí)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    uniq_mask = np.concatenate(([True], np.diff(x) != 0))
    x = x[uniq_mask]
    y = y[uniq_mask]

    return Spectrum(filename=uploaded_file.name, x=x, y=y)


def crop_spectrum(s: Spectrum, x_min: Optional[float], x_max: Optional[float]) -> Spectrum:
    """Ořízne spektrum podle zadaných mezí osy X."""
    mask = np.ones(s.x.shape, dtype=bool)
    if x_min is not None:
        mask &= (s.x >= x_min)
    if x_max is not None:
        mask &= (s.x <= x_max)
    
    if not np.any(mask):
        # Pokud by ořez smazal vše, vrátíme původní (nebo vyhodíme chybu)
        # Zde raději vrátíme původní s varováním v názvu, aby to nepadlo
        return s 

    return Spectrum(filename=s.filename, x=s.x[mask], y=s.y[mask])


def resample_spectrum(s: Spectrum, n_points: int) -> Spectrum:
    """Přeinterpoluje spektrum na n_points bodů v aktuálním rozsahu."""
    x_start = float(np.min(s.x))
    x_end = float(np.max(s.x))
    
    x_new = np.linspace(x_start, x_end, int(n_points), dtype=float)
    y_new = np.interp(x_new, s.x, s.y)
    return Spectrum(filename=s.filename, x=x_new, y=y_new)


def normalize_spectrum_max(s: Spectrum) -> Spectrum:
    """
    Vydělí celé spektrum jeho maximální hodnotou Y.
    Nejvyšší pík bude mít hodnotu 1.0.
    """
    max_val = np.max(s.y)
    
    # Ochrana proti dělení nulou (pokud je spektrum samé nuly)
    if max_val == 0:
        return s
    
    # Normalizace: y = y / max
    new_y = s.y / max_val
    return Spectrum(filename=s.filename, x=s.x, y=new_y)


def spectrum_to_txt_bytes(s: Spectrum, decimal: str = ".", sep: str = "\t") -> bytes:
    df = pd.DataFrame({"x": s.x, "y": s.y})
    buf = io.StringIO()
    # Format %.6f stačí, normalizovaná data jsou 0-1
    df.to_csv(buf, index=False, header=False, sep=sep, float_format="%.8f")
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
    st.set_page_config(page_title="Spectra Tool: Crop & Normalize", layout="centered")
    st.title("Spectra Processing Tool")
    st.markdown("""
    1. **Oříznutí (Crop)**: Odstraní okraje (např. laser na začátku), které kazí normalizaci.
    2. **Resampling**: Sjednotí počet bodů.
    3. **Normalizace**: Vydělí intenzity tak, že maximum = 1.
    """)

    # --- NASTAVENÍ ---
    with st.expander("Nastavení zpracování", expanded=True):
        st.subheader("1. Oříznutí osy X (důležité pro správnou normalizaci)")
        st.caption("Pokud máš na začátku spektra obrovský pík (laser), nastav 'Min X' až za něj (např. 150).")
        col_crop1, col_crop2 = st.columns(2)
        with col_crop1:
            crop_min = st.number_input("Min X (odkud začít)", value=0.0, step=10.0, help="Vše pod touto hodnotou bude smazáno.")
        with col_crop2:
            crop_max = st.number_input("Max X (kde skončit)", value=4000.0, step=50.0, help="Vše nad touto hodnotou bude smazáno. Nech 0 nebo velké číslo, pokud nechceš omezovat.")

        # Pokud uživatel nechce horní limit, interně použijeme None
        final_crop_max = crop_max if crop_max > crop_min else None

        st.divider()
        st.subheader("2. Resampling & Normalizace")
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            n_points = st.number_input("Počet bodů (n)", value=1500, step=100, min_value=10)
        with col_res2:
            do_normalize = st.checkbox("Normalizovat (Max = 1)", value=True)

        st.divider()
        st.subheader("3. Formát výstupu")
        col_fmt1, col_fmt2 = st.columns(2)
        with col_fmt1:
            decimal_out = st.selectbox("Desetinný oddělovač", [".", ","], index=0)
        with col_fmt2:
            sep_out_label = st.selectbox("Oddělovač sloupců", ["TAB", "Mezera", "Středník"], index=0)
            sep_out = "\t" if sep_out_label == "TAB" else (" " if sep_out_label == "Mezera" else ";")

    # --- UPLOAD ---
    files = st.file_uploader("Nahraj .txt spektra", type=["txt"], accept_multiple_files=True)

    if not files:
        st.info("Čekám na soubory...")
        return

    if st.button("Zpracovat soubory", type="primary"):
        try:
            processed_spectra = []
            
            for f in files:
                # 1. Načíst
                s = read_spectrum_txt(f)
                
                # 2. Oříznout (Crop) - KLÍČOVÝ KROK
                # Tím se zbavíme falešných maxim na krajích
                s = crop_spectrum(s, x_min=crop_min, x_max=final_crop_max)
                
                if len(s.x) < 2:
                    st.warning(f"Soubor {s.filename} byl ořezáním smazán celý (špatný rozsah?). Přeskakuji.")
                    continue

                # 3. Resample
                s = resample_spectrum(s, n_points)
                
                # 4. Normalizace
                if do_normalize:
                    s = normalize_spectrum_max(s)
                
                processed_spectra.append(s)

            if not processed_spectra:
                st.error("Žádná spektra k uložení (všechna byla ořezána na 0 bodů). Zkontroluj rozsahy Min/Max.")
                return

            # --- VÝSLEDEK ---
            zip_bytes = build_zip(processed_spectra, decimal_out, sep_out)
            
            st.success(f"Zpracováno {len(processed_spectra)} souborů.")
            
            st.download_button(
                label="Stáhnout ZIP",
                data=zip_bytes,
                file_name="spectra_normalized.zip",
                mime="application/zip"
            )

            # --- NÁHLED ---
            st.markdown("### Kontrola výsledku (první soubor)")
            first = processed_spectra[0]
            
            # Zobrazíme graf, aby bylo hned vidět, jestli je to OK
            chart_data = pd.DataFrame({"x": first.x, "y": first.y})
            st.line_chart(chart_data, x="x", y="y")
            
            # Info o max hodnotě (pro kontrolu, mělo by být 1.0)
            st.caption(f"Max hodnota v náhledu: {np.max(first.y):.4f}")

        except Exception as e:
            st.error(f"Chyba: {e}")

if __name__ == "__main__":
    main()
