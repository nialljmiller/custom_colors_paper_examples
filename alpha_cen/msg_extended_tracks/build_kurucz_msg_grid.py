#!/usr/bin/env python3
"""Build an MSG specgrid from the Kurucz2003all SEDs used by MESA Colors."""

import os
import re
import subprocess
import tempfile
from pathlib import Path

MSG = Path(os.environ["MSG_DIR"])
MESA = Path("/home/njm/MESA/mesa_filter_interp_leak_fix")
SOURCE = MESA / "data/colors_data/stellar_models/Kurucz2003all"
OUT = Path(__file__).resolve().parent / "kurucz2003_colors"
SPECINTS = OUT / "specints"
MANIFEST = OUT / "specints.list"
GRID = OUT / "sg-Kurucz2003all.h5"
CONVERTER = MSG / "bin/ascii_to_specint"
PACKAGER = MSG / "bin/specint_to_specgrid"


def metadata(path):
    text = path.read_text().splitlines()
    result = {}
    for line in text:
        m = re.match(r"#\s*(teff|logg|metallicity)\s*=\s*([-+\d.Ee]+)", line)
        if m:
            result[m.group(1)] = float(m.group(2))
    if set(result) != {"teff", "logg", "metallicity"}:
        raise ValueError(f"Missing atmosphere labels in {path}: {result}")
    return result


def main():
    if not SOURCE.is_dir() or not CONVERTER.is_file() or not PACKAGER.is_file():
        raise FileNotFoundError("MESA Colors Kurucz files or MSG grid tools are missing")
    OUT.mkdir(parents=True, exist_ok=True)
    SPECINTS.mkdir(parents=True, exist_ok=True)
    files = sorted(SOURCE.glob("Kurucz2003all_fid*.txt"))
    if len(files) != 3808:
        raise RuntimeError(f"Expected 3808 Kurucz SEDs, found {len(files)}")

    with tempfile.TemporaryDirectory(prefix="kurucz-msg-") as temp_dir:
        temp = Path(temp_dir)
        manifest_entries = []
        for i, source in enumerate(files, 1):
            x = metadata(source)
            ascii_file = temp / (source.stem + ".dat")
            specint = SPECINTS / (source.stem + ".h5")
            # Strip descriptive comments; MSG receives the axes explicitly.
            ascii_file.write_text("\n".join(line for line in source.read_text().splitlines()
                                           if line and not line.startswith("#")) + "\n")
            command = [str(CONVERTER), str(ascii_file), str(specint),
                       "-l", f"Teff:{x['teff']:.12g}",
                       "-l", f"log(g):{x['logg']:.12g}",
                       "-l", f"[Fe/H]:{x['metallicity']:.12g}"]
            subprocess.run(command, check=True, env=os.environ.copy(),
                           stdout=subprocess.DEVNULL)
            manifest_entries.append(str(specint))
            if i % 250 == 0 or i == len(files):
                print(f"Converted {i}/{len(files)} spectra", flush=True)

        MANIFEST.write_text("\n".join(manifest_entries) + "\n")
    command = [str(PACKAGER), str(MANIFEST), str(GRID),
               "--remove-orphans", "--compress-axes", "--grid-label=Kurucz2003all-MESA-Colors"]
    subprocess.run(command, check=True, env=os.environ.copy())
    print(f"Wrote {GRID}")


if __name__ == "__main__":
    main()
