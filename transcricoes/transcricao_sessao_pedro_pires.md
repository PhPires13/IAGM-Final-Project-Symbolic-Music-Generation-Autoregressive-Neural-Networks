# Session transcript "IAGM - Final Project (Pedro)"

DCC831 - Generative AI for Music (UFMG, 2026/2) - Final Project: Symbolic Music Generation with Autoregressive Neural Networks

Conversation between Pedro Pires and Claude (Claude Code, Opus 5.5, medium effort), from 11/19/2026 to 11/25/2026. Times in Brasília time.

---

## Pedro (11/19 16:48)

The document @tp2_autoregressivos.md has the requirements of my music final project

Make a plan of what needs to be done to meet all the requirements
Guide/teach me through each part of the project step by step with code examples for what I want to do, without doing it directly unless explicitly asked (Keep this in memory)

## Claude

I'll use the assignment (`tp2_autoregressivos.md`) as the base, so every item has a matching step. Since the deadline is 11/25, the plan always picks the simplest path and spends the effort where the grade is: meeting every item and doing a solid evaluation that compares distributions, not just means.

#### Final Project Plan — Symbolic Music Generation

##### 1. Initial decisions

| Decision | Proposal | Why |
|---|---|---|
| Task | **Free (unconditional) composition** | No conditioning means less code and fewer ways to fail |
| Dataset | **Nottingham** (~1000 folk tunes, melody + chords) | Small, trains in minutes on a GPU and sounds good |
| Plan B | **JSB Chorales** (`music21.corpus`) | Already installed, no download |
| Representation | **REMI** via `miditok` | Explicit bar and position tokens, good for metrical music |
| Model | **Small decoder-only Transformer** in plain PyTorch | The assignment requires native PyTorch |
| Baseline | **Order-n Markov chain over the same REMI tokens** | Clearly simpler, and the TP1 idea can be reused |
| Evaluation | `muspy` metrics + overlap area and KL (Yang & Lerch, 2020) | The assignment asks for ≥ 3 metrics and a comparison of distributions |

##### 2. Pipeline

1. **Data:** split **by song** (80/10/10 with a fixed seed, or the official one if it exists), saved to `splits.json`. Transposition from −5 to +6 semitones **only in training**.
2. **Tokenization:** REMI with `BOS`/`EOS`, 512-token windows with a 256 stride. Report vocabulary size and mean length.
3. **Model:** `nn.Embedding` + learned positional embedding + `nn.TransformerEncoderLayer` with a causal mask + linear head (~3–4M parameters).
4. **Training:** next-token cross-entropy, AdamW, warmup + cosine, gradient clipping, best checkpoint by validation loss. Log train and validation loss to a CSV for the paper figure.
5. **Sampling:** temperature, top-k and top-p written by the group; 200 samples per system.
6. **Baseline:** n-grams with backoff, order picked on validation, same data and same number of samples.
7. **Evaluation:** crop the test set to the same number of bars as the samples (metrics like pitch range depend on length), compute the metrics per piece and compare the distributions with OA and KL.
8. **Audio:** FluidSynth + a free soundfont, 5 seeded random samples per system in `samples/`.

##### 3. Deliverables

- [ ] Public repository with code, `README.md` (install, reproduction and each member's contribution) and `requirements.txt`
- [ ] Paper in the ISMIR template, up to 4 pages of content
- [ ] 5 MIDIs from the Transformer and 5 from the baseline, with audio, cited in the paper
- [ ] In-class presentation

**Up to the group:** the task split depends on the group size. With more than 3 people, it's worth separating data, model, baseline + evaluation, paper and audio/repository.

---

## Pedro (11/19 17:09)

we are 5

put this in a plan file with the task split so I can send it to the group

## Claude

Created `PLAN.md` at the root with:

1. **Project decisions:** the table above, plus an LSTM as an optional extra baseline (it reuses ~95% of the training loop) and a note about the extra requirements in case someone in the group is a grad student.
2. **Technical pipeline:** one subsection per step (data, tokenization, model, training, sampling, Markov, evaluation, audio), with starting hyperparameters and the layout of the paper's main table (mean ± std, OA and KL per system).
3. **Repository layout:** one file per step in `src/` and the command sequence that goes in the README.
4. **Task split:** five fronts (A – Data, B – Model, C – Baseline + Evaluation, D – Paper, E – Audio/Repository/Slides), each with the part of the paper it writes.

I used letters instead of names so the group can choose. Worth confirming the dataset with them before anyone starts writing code.

---

## Pedro (11/21 13:22)

the group changed the plan, it's going to be emopia with energy and tension curves now

I got the data part
how do I split by song? is there an official split?

## Claude

It does. EMOPIA 1.0 ships a notebook (`scripts/prepare_split.ipynb`) that produces the official split, and reproducing it beats inventing one: results stay comparable with the literature and it's already by song. Put dataset access in `src/data/emopia.py` and the split in `src/data/split.py`.

**1. Clip names.** Each MIDI is named `Q<quadrant>_<YouTube id>_<clip index>`, e.g. `Q1_0vLPYiPN7qY_0`. Some YouTube ids contain `_`, so **don't use `split("_")`**. The id is always 11 characters, so slice by position:
```python
def song_id(clip: str) -> str:
    return clip[3:14]

def quadrant(clip: str) -> int:
    return int(clip[1])
```

**2. Extraction.** Write an `ensure_extracted(cfg)` that unzips into `data/raw/` only the first time, skipping macOS junk (`__MACOSX/` and `.DS_Store`).

**3. Official split.** The notebook reads `metadata_by_song.csv`, samples **8 songs per dominant quadrant** for test and, from the rest, 8 more per quadrant for validation, always with `random_state=1`:
```python
def _sample_by_quadrant(songs):
    return pd.concat([songs[songs["DominantQ"] == q].sample(8, random_state=1) for q in (1, 2, 3, 4)])

test = _sample_by_quadrant(data)
train_val = data.drop(labels=test.index)
val = _sample_by_quadrant(train_val)
train = train_val.drop(labels=val.index)
```

**4. From songs to clips.** Then assign each clip to its song's split and save `{"train": [...], "val": [...], "test": [...]}`. An `assert` guarantees no song landed in two splits:
```python
song_to_split = {s: name for name, ss in songs.items() for s in ss}
assert len(song_to_split) == sum(len(ss) for ss in songs.values()), "song in more than one split"
```

I'd save it to `configs/splits.json` (versioned) rather than `data/`, which is gitignored, so everyone uses exactly the same split. Also print the clip count per quadrant in each split to check the balance.

---

## Pedro (11/21 14:06)

now the tokenizer

it needs to stay fixed after we create it and already have room for the control tokens

## Claude

Put it in `src/data/tokenizer.py`. The idea is to build REMI once, save it to `configs/tokenizer.json`, and from then on always load from that file.

**Config.** A few choices depend on the dataset:
- **Tempo and time signature off:** EMOPIA MIDIs are audio transcriptions and all declare 120 BPM and 4/4, so those tokens would carry no information.
- **Velocity on (32 levels):** Julia's energy uses mean velocity, so it has to survive tokenization.
- **Control tokens as special tokens:** `Energy_*` and `Tension_*` go in `special_tokens`. Use the contract 3 function (`control_token_names`, in `src/controls/tokens.py`) to build the names, so both sides never diverge even before Julia finishes her part.

```python
config = TokenizerConfig(
    pitch_range=(21, 109),               # piano keyboard
    beat_res={(0, 4): 8, (4, 12): 4},    # positions at 1/8 of a beat
    num_velocities=32,
    use_chords=False, use_rests=False,
    use_tempos=False, use_time_signatures=False, use_programs=False,
    special_tokens=["PAD", "BOS", "EOS", *control_token_names(n_levels)],
)
tokenizer = REMI(config)
tokenizer.save("configs/tokenizer.json")
```

**Actually freeze it.** Make `main` refuse to overwrite the file unless `--force` is passed. If someone regenerates the tokenizer without telling the group, the ids change and every checkpoint becomes invalid:
```python
if out.exists() and not args.force:
    raise SystemExit(f"{out} already exists and is frozen; use --force only if the group agrees")
```

**Load and convert.** Write `load_tokenizer()` (`REMI(params=path)`), `encode(tokenizer, score) -> list[int]` and `decode(tokenizer, ids) -> Score`. `encode` doesn't add `BOS`/`EOS` or controls; that's the dataset build's job.

The vocabulary should land around 290 tokens. Write the number down, it goes in the paper.

---

## Pedro (11/21 14:52)

how do I build the dataset with the controls in it

transposition only on train

## Claude

Write a `src/data/build_dataset.py` that goes clip by clip and saves one `.pt` per split.

**1. Measure on the quantized version.** Tokenize the clip and decode it back before measuring energy and tension with contract 4's `measure_piece` (Julia is still implementing it; for now it's Leo's stub). It looks redundant, but it guarantees two things: the measured bars are exactly the ones that become `Bar` tokens, and it's what will happen to the generated pieces at evaluation time.
```python
levels = to_levels(measure_piece(decode(tokenizer, encode(tokenizer, score))), thresholds)
```

**2. Transposition only in training.** In train, use the shifts from `data.transpose_range` (−5 to +6); in validation and test, only 0. Drop any shift that pushes a note off the keyboard (21 to 108):
```python
def _transpositions(score, shifts):
    pitches = [n.pitch for t in score.tracks for n in t.notes]
    lo, hi = min(pitches), max(pitches)
    for s in shifts:
        if 21 <= lo + s and hi + s <= 108:
            yield s, (score if s == 0 else score.shift_pitch(s))
```
The levels measured on the original hold for every transposition: energy doesn't depend on pitch, and tension is relative to the key, which moves along.

**3. Insert the controls.** Use contract 3's `insert_controls(ids, levels, vocab)`, which puts `Energy_e Tension_t` after each `Bar`, and wrap with `BOS` and `EOS`.

**4. Save.** One `torch.save` per split with the full sequences and the metadata evaluation will need:
```python
{"sequences": [...], "clip_ids": [...], "transpose": [...], "levels": [...], "controls": True, "vocab_size": len(tokenizer)}
```

**Don't wait for the tertiles.** `control_thresholds.json` depends on Julia. Add a `--no-controls` flag that skips measuring and inserting, so Natan can already test training on the real pipeline.

---

## Pedro (11/21 15:41)

and cutting it into windows for training?

wanted to keep that separate

## Claude

Keeping it separate is a good call: the `.pt` stores the whole piece, and the window size can change with the model without rebuilding the dataset. Put a `WindowDataset(Dataset)` class in `src/data/dataset.py`.

**Window index.** In `__init__`, walk the sequences and store `(sequence, start)` pairs. Each window has `max_len + 1` tokens (input and target shifted by one), with step `stride`. Add a last window that ends at the end of the sequence so the ending of the piece isn't lost:
```python
span = max_len + 1
starts = list(range(0, max(n - span, 0) + 1, stride))
if n > span and starts[-1] != n - span:
    starts.append(n - span)
```

**Items.** Sequences shorter than the window are padded with `PAD`. In the target, `PAD` positions get `-100`, which `cross_entropy` ignores:
```python
x = torch.full((max_len,), pad_id)
y = torch.full((max_len,), -100)
x[: len(chunk) - 1] = chunk[:-1]
y[: len(chunk) - 1] = chunk[1:]
```

Also add a `load_split(split)` that reads the `.pt`, so Natan doesn't need to know where the files live.

---

## Pedro (11/21 16:12)

what stats of the dataset should go in the paper

## Claude

The assignment asks for dataset statistics, vocabulary size and mean sequence length. Write a `src/data/stats.py` that computes, per split:

| Statistic | How |
|---|---|
| Songs and clips | `len({song_id(c) for c in clips})` and `len(clips)` |
| Clips per quadrant | `Counter(quadrant(c) for c in clips)` |
| Clip duration (s) | `score.to("second").end()` |
| Notes per clip | sum of `len(t.notes)` over tracks |
| Tokens per clip | only the **original** sequences (`transpose == 0`), otherwise augmentation inflates it |
| Sequences with augmentation and windows | `len(data["sequences"])` and `len(WindowDataset(...))` |

For each numeric value keep mean, std, min and max. Save to `outputs/stats/dataset_stats.json` and print a markdown table, which is almost ready for the paper.

Then add the data commands to the README:
```bash
uv run python -m src.data.split
uv run python -m src.data.tokenizer
uv run python -m src.data.build_dataset
uv run python -m src.data.stats
```

That closes contracts 2 and 5. Suggested commit: `data: official emopia split, remi tokenizer and windowed dataset`.

---

## Pedro (11/23 17:46)

first full training finished

the loss looks fine but someone in the group who plays listened and said it sounds random, no key and no rhythm
lowering the temperature barely helped

what could it be

## Claude

When perplexity is good and the sound is bad, suspect the data before the model. Two signs point that way:

**1. The bar may not be a bar.** Every EMOPIA 1.0 MIDI declares 120 BPM and 4/4. If that tempo is just the default value of an automatic transcription, the notes aren't aligned to any pulse. Then the "bar" the model sees is a 2-second window, the `Position` tokens are noise, and the energy and tension curves are measured over windows unrelated to the music.

**2. Overfitting.** Check `metrics.csv` for training loss still dropping while validation loss rises. With few clips, a 19.6M-parameter model memorizes the training set in a few epochs.

To test the first hypothesis, measure the fraction of note onsets that fall near the sixteenth-note grid. If the notes are aligned, almost all fall near it; if not, you get roughly chance (10% with a 5% tolerance on each side):
```python
score = symusic.Score(path)
q = score.ticks_per_quarter / 4                 # one sixteenth in ticks
for n in score.tracks[0].notes:
    frac = (n.time / q) % 1.0
    near += min(frac, 1 - frac) < 0.05
```

Run it on the whole training set and compare with a dataset you know is quantized.

---

## Pedro (11/23 18:24)

it's basically random on emopia
natan said the big model overfit really fast too

found emopia+ and pop1k7 from the same lab, emopia+ looks way better on that test, pop1k7 a bit less

what can we do with 2 days left

## Claude

That confirms it: the problem is the data. EMOPIA's own authors don't generate from those raw MIDIs; they sync them to the audio beat and pretrain on a larger set.

| Dataset | What it is | Verdict |
|---|---|---|
| **EMOPIA+** | The same 1071 clips, same names and same split, beat-quantized and split into tracks | **Replaces EMOPIA 1.0** without changing the experimental design |
| **Pop1K7** | 1747 pop piano covers (~108 h, 12× EMOPIA), beat-synchronized | **Pretraining.** Same genre and instrument. It scores lower because of expressive timing *within* the beat; the beats themselves are aligned |
| MAESTRO, GiantMIDI | Live classical piano, no grid | Would repeat the rhythm problem, and it's another genre |
| POP909 | Multi-track pop arrangements | Smaller than Pop1K7 and a different format |

**Suggested design, in two phases:**
1. **Pretrain on Pop1K7**, with energy and tension tokens measured by the same function: the model learns key, progressions and meter.
2. **Fine-tune on EMOPIA+** from those weights, with a lower learning rate (1e-4).

This hits both problems: EMOPIA+ fixes the rhythm and Pop1K7 adds volume. As a bonus, EMOPIA has emotion labels: an `Emotion_Q1..Q4` token right after `BOS` (and `Emotion_None` on Pop1K7) turns the task into conditioning on emotion **and** curves, without changing the architecture.

**Two cautions:**
- **Leakage:** both sets are YouTube covers collected by the same lab, so an EMOPIA+ test song may be in Pop1K7. That has to be checked before pretraining.
- **Fair baseline:** the Markov chain has to be fit on the same data the Transformer sees, i.e. Pop1K7 + EMOPIA+.

Keep the same contracts where possible (measurement, model interface, evaluation curves), so the rest of the group changes as little as possible.

---

## Pedro (11/23 18:57)

group agreed

write a doc about this change so everyone follows it and we can use it in the paper
leave what we don't know yet blank

## Claude

Created `docs/pivo_pretreino.md`, organized so it can become the method and discussion sections of the paper (written in Portuguese, like the rest of the group docs):

1. **Motivation:** what went wrong in the first training, with the alignment measurements from your script (9.5% / 91.3% / 31.6%) and the consequences (fake bar, `Position` as noise, overfitting in 4 epochs).
2. **Data choice:** the dataset table above, with each reference and Zenodo link.
3. **Two-phase design:** pretraining and fine-tuning.
4. **Conditioning:** the sequence format, `BOS Emotion_Qx Bar Energy_e Tension_t <notes> ...`, and the rule that the emotion token never leaves the context (every window gets `BOS Emotion` in front).
5. **Leakage:** the checking method, with the number of removed songs _a preencher_ (to be filled).
6. **Baseline on the same data.**
7. **Experiments:** pretraining, fine-tuning and the from-scratch ablation, in both sizes, plus Markov; and the new per-quadrant emotion indicator (mean energy for arousal, fraction in major mode for valence).
8. **What changes in each roadmap contract:** a "before × now" table so each front knows exactly what to adjust.
9. **Limitations** and **references**.

The _a preencher_ fields (perplexities, removed songs) come from the cluster run. Worth telling Julia M. the tertiles must be recomputed on EMOPIA+, and Natan that the pretraining and fine-tuning configs are new.

---

## Pedro (11/24 07:41)

how do I adapt the data part to emopia+? it comes with separate tracks

## Claude

EMOPIA+ splits each clip into `Melody`, `Texture`, `Bass` and `Chord`. `Chord` is a chord annotation, not part of the performance, so drop it and merge the other three into a single piano track. Open a MIDI first to check; some clips have no `Texture`.

**1. Merge the tracks** (`src/data/emopia.py`):
```python
def merge_piano(score, skip=frozenset()):
    out = symusic.Score(score.ticks_per_quarter)
    out.tempos = score.tempos                      # keep the beat tempo map
    out.time_signatures = score.time_signatures
    piano = symusic.Track("piano", 0, False)
    for track in score.tracks:
        if track.name not in skip:                 # skip={"Chord"}
            piano.notes.extend(track.notes)
    piano.notes.sort(key=lambda n: (n.time, n.pitch))
    out.tracks.append(piano)
    return out
```
Keep `skip` as a parameter: Pop1K7 has a single track and will reuse the same function.

**2. Split.** EMOPIA+ already ships CSVs with the official split. Read them instead of the notebook, and check that each split's songs are exactly the ones from your earlier reproduction. If they are, that's worth a sentence in the paper.

**3. Reproducible download.** Both sets are CC-BY 4.0 on Zenodo. A `scripts/download_data.sh` that downloads and checks the size prevents anyone from training on a corrupted zip:
```bash
fetch() {
    local file="$1" url="$2" size="$3"
    curl -fL --retry 3 -o "$file.part" "$url"
    mv "$file.part" "$file"
    [ "$(stat -c %s "$file")" = "$size" ] || { echo "unexpected size for $file" >&2; exit 1; }
}
```

---

## Pedro (11/24 08:16)

and pop1k7? don't want to duplicate everything

## Claude

The trick is to treat Pop1K7 as more splits in the same `splits.json`, with a name prefix: `pretrain_train` and `pretrain_val`.

**1. `src/data/pop1k7.py`:** extract only `midi_synchronized/` (the zip has other versions you don't need) and identify each song as `src_00k/<n>`. Hold out 5% of the songs for pretraining validation.

**2. `src/data/corpus.py`:** a single function that picks the source from the split name. The build, Julia's calibration and the stats all go through it:
```python
def is_pretrain(split):
    return split.startswith("pretrain_")

def load_piece(split, item, roots):
    if is_pretrain(split):
        return pop1k7.load_score(roots["pop1k7"], item), None          # no emotion
    return emopia.load_score(roots["emopia"], item), emopia.quadrant(item)
```
A `None` quadrant becomes `Emotion_None`.

**3. Config.** In `default.yaml`, add a `pretrain` section with `val_fraction: 0.05` and `transpose_range: [-3, 3]`. The range can be smaller than EMOPIA's (−5 to +6) because Pop1K7 is already 12× larger and the dataset would explode.

**4. Emotion in every window.** Add `keep_prefix: 2` under `data`. In `WindowDataset`, split each sequence into prefix (`BOS Emotion`, the first k tokens) and body, window only the body in chunks of `max_len + 1 - k`, and glue the prefix in front of **every** window. That way emotion never leaves the context, just like at generation time:
```python
span = max_len + 1 - keep_prefix
n = len(seq) - keep_prefix                 # body length
starts = list(range(0, max(n - span, 0) + 1, stride))
...
chunk = torch.cat([seq[:k], seq[k + start : k + start + span]])
```
With `keep_prefix = 0` it's the plain windowing from before.

---

## Pedro (11/24 08:52)

is the tokenizer grid still ok or should I change it

## Claude

It depends on the data, and now that notes are on the beat you can measure instead of guessing. Repeat yesterday's measurement with several grids:
```python
for div in (4, 8, 12):
    q = score.ticks_per_quarter / div
    ...  # fraction of onsets within 5% of a grid point
```

What I'd expect: the 1/8 grid adds nothing over 1/4 (anything on 1/8 is already on 1/4), and the 1/12 grid catches triplets, which 1/4 misses. If that's what you get, switch to 12 positions per beat:
```python
beat_res={(0, 4): 12, (4, 12): 4},   # positions at 1/12 of a beat (sixteenths and triplets)
```

Since this changes the vocabulary, the tokenizer has to be regenerated with `--force`, which is exactly the case the group needs to agree on. Use the same change to reserve the `Emotion_Q1..Q4` and `Emotion_None` tokens. Also update the docstring: tempo now varies beat by beat (it comes from audio sync), and rendering uses a fixed tempo, the same for human and generated pieces.

---

## Pedro (11/24 09:18)

1/12 catches more, changed it

how do I check if a song from pop1k7 is also in emopia? key and tempo can be different

## Claude

Use a fingerprint that is invariant to transposition and tempo, and measure **containment**: what fraction of the clip appears inside the song. Put it in `src/data/leakage.py`.

**1. Top line.** Covers change the accompaniment but rarely the melody. Take the highest note of each onset, grouping onsets on a 1/4-beat grid:
```python
def top_line(score):
    q = max(1, score.ticks_per_quarter // 4)
    top = {}
    for track in score.tracks:
        for n in track.notes:
            slot = round(n.time / q)
            if n.pitch > top.get(slot, -1):
                top[slot] = n.pitch
    return [top[k] for k in sorted(top)]
```

**2. Intervals and n-grams.** Use the intervals between consecutive notes (transposition-invariant) and their set of n-grams. Ignoring durations makes it tempo-invariant. Start with n = 6.
```python
iv = np.diff(np.array(top_line(score)))
fp = {tuple(iv[i : i + n]) for i in range(len(iv) - n + 1)}
```

**3. Containment.** For a clip C and a song S: `|ngrams(C) ∩ ngrams(S)| / |ngrams(C)|`. Brute-forcing 200 clips against 1747 songs is slow; an inverted index (n-gram → songs) makes each query touch only the songs that share some n-gram.

**4. Calibrated threshold.** Don't guess it: compute the same containment between validation/test clips and EMOPIA+ **training** songs, which are different songs by construction. The threshold has to sit well above that distribution's p95. Something like 0.3 is usually safe.

**5. Output.** Remove from pretraining every song whose containment of some validation or test clip is above the threshold, and save the report to `configs/pop1k7_leakage.json` (versioned) with each removed song's containments.

Suggested commit: `data: beat-quantized emopia+, pop1k7 pretraining and leakage check`.

---

## Pedro (11/24 21:52)

data jobs and markov finished on the cluster

fill in the numbers in the pivot doc

## Claude

Read the logs and updated `docs/pivo_pretreino.md`:

- **Data size:** a new table with pieces and tokens per split. EMOPIA+ train: 869 clips, 9.9M tokens with augmentation, 975 per clip on average. Pop1K7: 1657 songs, 57.4M tokens (~8.2M original), 4957 on average.
- **Vocabulary:** 326 tokens (5 emotion tokens plus the positions and durations of the 1/12 grid, 48 positions per bar instead of 32).
- **Levels:** Julia's recomputed tertiles (energy 0.360 and 0.497; tension 0.067 and 0.145) give 33/33/33 levels, with a Spearman of 0.15 between energy and tension. Mean energy separates the arousal quadrants (Q1 0.51 and Q2 0.57 against Q3 and Q4 0.35).
- **Leakage:** calibration between different songs gave a median of 0.015 and a p95 of 0.091. **3 Pop1K7 songs were removed**; two of them contain 51% to 66% of the n-grams of two validation/test songs, i.e. they are the same cover.
- **Markov:** 18.2M tokens, n = 4, D = 0.9, validation perplexity 25.7. I noted that it can't be compared directly with the earlier 32.7 because the vocabulary changed.

Two findings I recorded as limitations:
1. Calibration found a validation/test clip with 0.85 containment in an EMOPIA+ **training** song with a different YouTube id, probably two covers of the same composition. It's a property of the official split, which we keep to stay comparable with the literature.
2. On the human test clips themselves, major mode separates Q2 (16%) well from Q1 and Q4 (88% and 78%), but Q3 is still at 48%. Worth keeping in mind when Julia B. reads the valence indicator.

The Transformer perplexities are still _a preencher_ because training is still running.

---

## Pedro (11/25 09:37)

make a short pdf summary of the project to send to the group and the professor

and add a checklist of everything the assignment asks to the plan

## Claude

Done:

**PDF summary** (`entregas/IAGM_Projeto_Final_Resumo.pdf`, 1 page, in Portuguese for the professor): task (composition conditioned on emotion and on energy and tension curves), data (EMOPIA+ and Pop1K7, split by song, leakage check), representation (REMI 1/12 with control tokens), model and baseline, evaluation protocol and what's still missing.

**`PLAN.md`:**
- a note at the top about the pivot, linking `docs/pivo_pretreino.md`, `docs/pipeline.md` and `docs/notas_resultados.md`;
- the schedule, which was embedded content that didn't render on GitHub, is now a table with phases and milestones;
- the delivery checklist, split into **No artigo**, **No repositório** and **No Moodle**, one item per assignment requirement. The ones most often forgotten: split **by song** with test used only in the final evaluation, Transformer and sampling in native PyTorch without Hugging Face, baseline with the same data and protocol, comparison of **distributions** with 100+ samples, training and validation curves, and each member's contribution in the README.

---

## Pedro (11/25 12:07)

need audio for the samples

don't want to install fluidsynth on the mac, any other way?

## Claude

macOS already ships a GS sound bank (`gs_instruments.dls`), and `AVAudioEngine` can play a MIDI through it in offline mode, writing straight to a file. That fits in a small Swift script compiled with `swiftc`, which comes with the Xcode command line tools.

**`scripts/render_midi_macos.swift`:**
```swift
let engine = AVAudioEngine()
let sampler = AVAudioUnitSampler()
engine.attach(sampler)
engine.connect(sampler, to: engine.mainMixerNode, format: nil)
try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: 4096)
try sampler.loadSoundBankInstrument(at: soundBank, program: 0, bankMSB: 0x79, bankLSB: 0)  // acoustic piano
```
Then load the MIDI into an `AVAudioSequencer` attached to the engine, render in 4096-frame blocks until the end of the sequence plus the release, and write to a WAV `AVAudioFile`.

**In `render.py`**, add `--backend {fluidsynth,macos}`, keeping FluidSynth as the default for Linux. In the `macos` backend, compile the script once into a temp folder and convert the WAV to M4A (AAC) with `afconvert`, which is also built in:
```python
subprocess.run(["swiftc", "-O", str(MACOS_SCRIPT), "-o", str(binary)], check=True)
...
subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", "160000", str(wav), str(out)], check=True)
```

Check at startup that the chosen backend's tools exist (`shutil.which`) so the error is clear.

---

## Pedro (11/25 12:36)

worked

but it picks different ones for each model, I want to compare them on the same input

## Claude

Draw by **request** (curve + quadrant of a test clip) instead of by file. Each `manifest.json` already records each sample's `curve_clip`, so:

1. build, per system, a `curve_clip → sample` dict;
2. keep only requests present in **every** system;
3. draw `k` of them with the fixed seed and copy the matching sample from each system.

```python
by_clip = {s: {x["curve_clip"]: x for x in m} for s, (_, m) in manifests.items()}
common = sorted(set.intersection(*(set(c) for c in by_clip.values())))
chosen = sorted(random.Random(SEED).sample(common, args.k))
```

That way `finetune_large_014` and `markov_014` get the same request. The human clip has a different number in its manifest, so save a `samples/selection.json` that maps each file to its clip (`curve_clip`), with the emotion and requested levels. It'll serve the README and the paper's qualitative analysis.

Update the docstring to explain the pairing, so nobody thinks the samples were cherry-picked.

---

## Pedro (11/25 13:28)

now plan the paper in the ismir format

## Claude

Use the official ISMIR template in `latex/` and compile with `tectonic`. It's at most **4 pages of content**; references don't count. The paper stays in Portuguese, like the course.

**Structure** (items a–g of the assignment):
- **Abstract:** task, model, baseline and the headline numbers (mean OA, curve fidelity and the effect of pretraining).
- **Introduction:** why control how a piece evolves over time; the task (attribute-conditioned composition) and the contributions.
- **Related work:** Music Transformer, Pop Music Transformer (REMI), Compound Word Transformer (Pop1K7), EMOPIA and EMOPIA+, per-bar control (MuseMorphose, FIGARO), tonal tension (Herremans & Chew) and Yang & Lerch.
- **Method:** paragraphs with bold run-in labels: **Dados** (the pivot in two sentences, plus leakage), **Energia e tensão**, **Representação** (326-token vocabulary, 975 tokens per clip), **Modelo**, **Baseline**, **Geração e amostragem**.
- **Experiments:** hardware, sampling choice on validation, 100 pieces per system on test and the three evaluated aspects (similarity, fidelity, emotion).
- **Results and discussion:** three tables (perplexity; OA; fidelity with requested × shuffled curve) and three figures (training curves, violins, requested × obtained curve).
- **Conclusion:** findings, limitations and future work.

**Figures.** Generate them from `resultados_artigo/` with a script instead of copying PNGs from the cluster, so size and font match the ISMIR column:
```python
plt.rcParams.update({"font.size": 7, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(3.3, 1.55))
...
fig.savefig(OUT / "treino.pdf", bbox_inches="tight")
```

Put it in `latex/figuras/make_figures.py`, with the references in `latex/referencias.bib`.

---

## Pedro (11/25 14:33)

first version is in the latex with everyone's parts

improve the writing to sound like a paper, keep it in 4 pages

## Claude

Revised it keeping everyone's content. Main changes:

**Tone and cohesion**
- The text had five different voices. I made it consistently first person plural ("usamos", "propomos"), which is standard in multi-author papers.
- The pivot explanation showed up in the Introduction, the Method and the Conclusion. I kept two sentences in **Dados** and the lesson in the Conclusion ("a qualidade dos dados importa mais que o modelo").

**Accuracy**
- Numbers in the abstract, tables and text now match `resultados_artigo/`: mean OA of 0.927 against the 0.940 ceiling from human validation and 0.622 for Markov; Spearman of 0.81 (energy) and 0.48 (tension), dropping to zero with shuffled curves; and pretraining taking tension from 0.20 to 0.48.
- In the OA table, the small model (0.929) is slightly above the large one (0.927). I kept both in bold so the table doesn't suggest a difference that isn't there.
- The fidelity caption now says that with K = 3 chance gives about 0.33 exact accuracy, so readers can read the table on its own.

**Length**
- Merged repeated sentences and trimmed related work to what we use or compare with. Content fits in 4 pages.

**LaTeX**
- Decimal commas in math mode need braces, otherwise LaTeX adds a space: `$\beta=(0{,}9;\,0{,}95)$`.
- Thin space between number and unit: `19,6\,M`, `32\,min`.
- Replace "Proceedings of ISMIR" in the header with the course name, so it doesn't look published in the proceedings.

Worth having each person reread their own part to check the description is still right.

---

## Pedro (11/25 15:19)

check if the paper covers everything the assignment asks

## Claude

Checked item by item. The paper meets all of them:

| Requirement | Met? | Where |
|---|---|---|
| Task stated | Yes | Introduction: composition conditioned on emotion and curves |
| Dataset, preprocessing, statistics and split by song | Yes | Method, **Dados**: official split 869/114/88, test used only at the end |
| Representation justified, vocabulary and mean length | Yes | **Representação**: REMI because of the `Bar` token, 326 tokens, 975 per clip |
| How the condition is represented | Yes | Forced `Emotion`, `Energy` and `Tension` tokens |
| Native PyTorch Transformer, training and sampling by the group | Yes | **Modelo** and **Geração e amostragem** |
| Baseline with same data, representation and protocol | Yes | **Baseline**: Markov n = 4 on the same 18.2M tokens |
| ≥ 3 metrics, distributions, ≥ 100 samples | Yes | 7 MusPy metrics, OA and KL, 100 pieces per system, violins |
| Training and validation curves | Yes | Training figure |
| Qualitative analysis citing the repository pieces | Yes | Requested × obtained curve for piece `000` and the `samples/` pieces |
| ISMIR template, up to 4 pages | Yes | Content ends on page 4; references after |
| Structure (a)–(g) | Yes | All sections present |

Recompiled and copied the PDF to `entregas/IAGM_Projeto_Final_Artigo.pdf`. Add the LaTeX auxiliary files (`*.aux`, `*.log`, `*.bbl`) to `.gitignore` so they don't end up in the repo.

---

## Pedro (11/25 16:47)

the repo is a mess

organize it for submission, docs together, remove duplicates and make the readme easy to grade

## Claude

Organized:

- **`docs/`:** moved `tp2_autoregressivos.md` to `docs/enunciado.md` and `PLAN.md` to `docs/roadmap.md` with `git mv` to keep history. Fixed the links pointing to the old paths.
- **Duplicates:** `resultados_artigo/` had copies of `LEIA-ME_notas_resultados.md`, `pivo_pretreino.md`, `control_thresholds.json` and `pop1k7_leakage.json`, which already live in `docs/` and `configs/`. Removed the copies and left a short `resultados_artigo/README.md` pointing to the originals.
- **LaTeX:** removed `cc_by.eps` and `cc_by.png`; the template only uses `cc_by.pdf`.
- **README:** the **Resultados, amostras e artigo** section moved right after the description, with the PDF link and the table of the 5 drawn requests (`finetune_large_014`, `markov_014`, `human_014` …). The folder layout went to the end, one line per folder.

Searched for the old paths (`PLAN.md`, `tp2_autoregressivos.md`, `LEIA-ME`) and nothing broken is left. Open the README on GitHub after pushing to check the tables and links.
