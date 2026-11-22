"""Sorteia amostras de cada sistema, copia os MIDIs para `samples/` e renderiza áudio em MP3.

As amostras são sorteadas com seed fixa (não escolhidas a dedo). O áudio é renderizado com o
FluidSynth e um soundfont livre (ex.: FluidR3_GM ou GeneralUser GS) e convertido para MP3 com o
ffmpeg, para não pesar o repositório.

Uso:
  uv run python -m src.generate.render --soundfont /caminho/FluidR3_GM.sf2 transformer markov human
Saída: samples/<sistema>/<arquivo>.mid e .mp3 (versionados), mais samples/selection.json

Responsável: Leonardo.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.common import ROOT, load_config, resolve


def render(midi: Path, mp3: Path, soundfont: Path, sample_rate: int = 44100) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "out.wav"
        subprocess.run(["fluidsynth", "-ni", "-F", str(wav), "-r", str(sample_rate), str(soundfont), str(midi)],
                       check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-codec:a", "libmp3lame",
                        "-qscale:a", "4", str(mp3)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("systems", nargs="+", help="pastas em outputs/samples/")
    parser.add_argument("--soundfont", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--k", type=int, default=5, help="amostras por sistema")
    args = parser.parse_args()

    for tool in ("fluidsynth", "ffmpeg"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} não encontrado no PATH")
    cfg = load_config()
    soundfont = resolve(args.soundfont)
    rng = random.Random(cfg["seed"])
    selection = {}

    for system in args.systems:
        src_dir = resolve(cfg["paths"]["outputs_dir"]) / "samples" / system / args.split
        manifest = json.loads((src_dir / "manifest.json").read_text(encoding="utf-8"))
        chosen = sorted(rng.sample(manifest["samples"], args.k), key=lambda s: s["file"])
        dst = ROOT / "samples" / system
        dst.mkdir(parents=True, exist_ok=True)
        for s in chosen:
            stem = f"{system}_{Path(s['file']).stem}"
            shutil.copy(src_dir / s["file"], dst / f"{stem}.mid")
            render(dst / f"{stem}.mid", dst / f"{stem}.mp3", soundfont)
            print(f"  {dst / stem}.mid/.mp3 (curva de {s['curve_clip']})")
        selection[system] = [{"file": f"{system}_{Path(s['file']).stem}", "curve_clip": s["curve_clip"],
                              "requested_levels": s["requested_levels"]} for s in chosen]

    (ROOT / "samples" / "selection.json").write_text(json.dumps(selection, indent=1) + "\n", encoding="utf-8")
    print(f"seleção em {ROOT / 'samples' / 'selection.json'}")


if __name__ == "__main__":
    main()
