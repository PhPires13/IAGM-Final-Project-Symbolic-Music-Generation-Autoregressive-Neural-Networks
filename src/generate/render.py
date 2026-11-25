"""Sorteia amostras de cada sistema, copia os MIDIs para `samples/` e renderiza o áudio.

As amostras são sorteadas com seed fixa (não escolhidas a dedo) e **pareadas**: o mesmo pedido
(curva e quadrante de um clipe do teste) é usado em todos os sistemas. Os sistemas gerados
compartilham a numeração (`finetune_large_014` e `markov_014` receberam o mesmo pedido); o clipe
humano correspondente tem outra numeração, e o `samples/selection.json` liga cada arquivo ao clipe
(`curve_clip`).

Dois backends de áudio:
  - `fluidsynth` (padrão): FluidSynth + soundfont livre (ex.: FluidR3_GM), convertido para MP3
    com o ffmpeg;
  - `macos`: AVAudioEngine com o banco GS do próprio macOS (`scripts/render_midi_macos.swift`),
    convertido para M4A (AAC) com o `afconvert`. Não precisa de soundfont nem de ffmpeg.

Uso:
  python -m src.generate.render --soundfont /caminho/FluidR3_GM.sf2 finetune_large markov human
  python -m src.generate.render --backend macos --samples-dir resultados_artigo/musicas/teste \
      finetune_large markov human
Saída: samples/<sistema>/<sistema>_<nnn>.mid e .mp3/.m4a (versionados), mais samples/selection.json

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

ROOT = Path(__file__).resolve().parents[2]
SEED = 42  # mesma seed de configs/default.yaml
MACOS_SCRIPT = ROOT / "scripts" / "render_midi_macos.swift"


def render_fluidsynth(midi: Path, out: Path, soundfont: Path, sample_rate: int = 44100) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "out.wav"
        subprocess.run(["fluidsynth", "-ni", "-F", str(wav), "-r", str(sample_rate), str(soundfont), str(midi)],
                       check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-codec:a", "libmp3lame",
                        "-qscale:a", "4", str(out)], check=True)


def render_macos(midi: Path, out: Path, binary: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "out.wav"
        subprocess.run([str(binary), str(midi), str(wav)], check=True, capture_output=True)
        subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", "160000", str(wav), str(out)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("systems", nargs="+", help="pastas com manifest.json dentro de --samples-dir")
    parser.add_argument("--samples-dir", default="outputs/samples",
                        help="pasta com uma subpasta por sistema (padrão: saída do pipeline)")
    parser.add_argument("--split", default="test", help="subpasta da partição (vazio se não houver)")
    parser.add_argument("--backend", choices=["fluidsynth", "macos"], default="fluidsynth")
    parser.add_argument("--soundfont", help="obrigatório com --backend fluidsynth")
    parser.add_argument("--k", type=int, default=5, help="amostras por sistema")
    args = parser.parse_args()

    tools = ("fluidsynth", "ffmpeg") if args.backend == "fluidsynth" else ("swiftc", "afconvert")
    for tool in tools:
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} não encontrado no PATH")
    if args.backend == "fluidsynth" and not args.soundfont:
        raise SystemExit("--soundfont é obrigatório com --backend fluidsynth")

    base = (ROOT / args.samples_dir).resolve()
    manifests = {}
    for system in args.systems:
        d = base / system / args.split if args.split and (base / system / args.split).is_dir() else base / system
        manifests[system] = (d, json.loads((d / "manifest.json").read_text(encoding="utf-8"))["samples"])

    # sorteio pareado: só pedidos (clipes de curva) presentes em todos os sistemas
    by_clip = {s: {x["curve_clip"]: x for x in m} for s, (_, m) in manifests.items()}
    common = sorted(set.intersection(*(set(c) for c in by_clip.values())))
    chosen = sorted(random.Random(SEED).sample(common, args.k))

    with tempfile.TemporaryDirectory() as tmp:
        if args.backend == "macos":
            binary = Path(tmp) / "render_midi"
            subprocess.run(["swiftc", "-O", str(MACOS_SCRIPT), "-o", str(binary)], check=True)
        ext = ".mp3" if args.backend == "fluidsynth" else ".m4a"
        selection = {}
        for system, (src_dir, _) in manifests.items():
            dst = ROOT / "samples" / system
            dst.mkdir(parents=True, exist_ok=True)
            selection[system] = []
            for clip in chosen:
                s = by_clip[system][clip]
                stem = f"{system}_{Path(s['file']).stem}"
                shutil.copy(src_dir / s["file"], dst / f"{stem}.mid")
                if args.backend == "fluidsynth":
                    render_fluidsynth(dst / f"{stem}.mid", dst / f"{stem}{ext}", Path(args.soundfont))
                else:
                    render_macos(dst / f"{stem}.mid", dst / f"{stem}{ext}", binary)
                print(f"  {dst / stem}.mid/{ext} (curva de {clip})")
                selection[system].append({"file": stem, "curve_clip": clip, "emotion": s.get("emotion"),
                                           "requested_levels": s.get("requested_levels")})

    (ROOT / "samples" / "selection.json").write_text(json.dumps(selection, indent=1) + "\n", encoding="utf-8")
    print(f"seleção em {ROOT / 'samples' / 'selection.json'}")


if __name__ == "__main__":
    main()
