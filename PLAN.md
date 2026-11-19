# Planejamento – TP2: Geração Musical Simbólica com Redes Autoregressivas

**Entrega:** 25/11/2026, 23:59h · **Hoje:** 20/11 → temos ~5 dias. O plano foi feito para caber nisso.

Princípio: **escolher o caminho mais simples em cada decisão** e gastar o esforço onde a nota está — cumprir *todos* os itens do enunciado e fazer uma avaliação bem feita (distribuições, não só médias).

---

## 1. Decisões do projeto (resumo)

| Item | Escolha | Por quê |
|---|---|---|
| Tarefa | **Composição livre (incondicional)**, a partir do zero (+ continuação de *prompt* como análise qualitativa opcional) | Não precisa de condicionamento → menos código e menos chance de erro |
| Dataset | **Nottingham** (~1000 músicas folclóricas, melodia + acordes) | Pequeno, treina em minutos numa GPU do Colab, gera resultados agradáveis de ouvir |
| Plano B de dataset | **JSB Chorales** (`music21.corpus`, já vem instalado, sem download) | Usar se o Nottingham der problema no dia 1 |
| Representação | **REMI** via `miditok` | Codifica compasso e posição na batida explicitamente → bom para música métrica com ritmo regular |
| Modelo | **Transformer decoder-only pequeno** em PyTorch puro (`nn.TransformerEncoderLayer` + máscara causal) | Exigido: PyTorch nativo; Transformer dá o melhor resultado |
| Baseline | **Cadeia de Markov de ordem n sobre os mesmos tokens REMI** (reaproveitar o TP1) | Claramente “mais simples”; já temos código |
| Baseline extra (opcional) | LSTM usando o mesmo laço de treino | Só se sobrar tempo; reaproveita ~95% do código |
| Avaliação | 5 métricas do `muspy` + *overlap area* e KL (Yang & Lerch, 2020) | Enunciado pede ≥3 métricas e comparação de distribuições |

> ⚠️ Se o grupo tiver **algum aluno de pós-graduação**, há requisitos extras (≥5 artigos recentes nos trabalhos relacionados + experimento adicional, ex.: estudo de temperatura/top-p). Neste caso a sugestão mais barata é a **ablação de amostragem** (temperatura ∈ {0.8, 1.0, 1.2}), que reaproveita todo o pipeline.

---

## 2. Pipeline técnico

### 2.1 Dados (`src/data.py`)
1. Baixar os MIDIs do Nottingham (repositório `jukedeck/nottingham-dataset` no GitHub, pasta `MIDI/`; alternativa: `muspy.NottinghamDatabase`). Conferir no dia 1 se baixa e abre sem erro; se não, trocar para JSB Chorales.
2. **Divisão por música** (não por trecho), 80/10/10, com `seed` fixa, salva em `data/splits.json`. Se existir partição oficial, usar a oficial.
3. Pré-processamento:
   - mesclar todas as trilhas em **uma única trilha** (melodia + acordes) antes de tokenizar — evita complicação com múltiplos instrumentos;
   - descartar músicas muito curtas (ex.: < 8 compassos) e as que não forem 4/4 ou 3/4 se der problema;
   - *data augmentation*: transposição de −5 a +6 semitons **apenas no treino**.
4. Estatísticas para o artigo: nº de músicas por partição, duração média, nº médio de notas, faixa de alturas.

### 2.2 Tokenização (`src/tokenize.py`)
- `miditok.REMI` com configuração padrão (Bar, Position, Pitch, Velocity, Duration). Pode desligar Velocity se o dataset tiver velocity constante (reduz o vocabulário).
- Adicionar tokens `BOS`/`EOS`.
- **Reportar no artigo:** tamanho do vocabulário e comprimento médio (± desvio) das sequências.
- Para treinar: recortar em janelas de 512 tokens (com *stride* de 256).

### 2.3 Modelo (`src/model.py`) – escrito pelo grupo
```
Embedding(vocab, 256) + embedding posicional aprendido (512)
→ 4 × nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, dropout=0.1, batch_first=True)
  com máscara causal (nn.Transformer.generate_square_subsequent_mask)
→ LayerNorm → Linear(256, vocab)
```
~3–4M parâmetros: suficiente para o Nottingham e treina rápido.

### 2.4 Treino (`src/train.py`) – escrito pelo grupo
- Perda: *cross-entropy* do próximo token (*teacher forcing*).
- AdamW, lr 3e-4, *warmup* curto + *cosine decay*, batch 32, gradiente clipado em 1.0.
- ~30 épocas, salvar o melhor checkpoint pela **perda de validação** (*early stopping*).
- Logar perda de treino e validação por época em CSV → vira a **figura de curvas de treino** do artigo.
- Rodar no Google Colab (GPU T4 gratuita) ou no Mac (`device="mps"`).

### 2.5 Amostragem (`src/sample.py`) – escrita pelo grupo
- Laço autoregressivo token a token com **temperatura + top-k + top-p** (implementar os três; são ~20 linhas).
- Configuração padrão: temperatura 1.0, top-p 0.95.
- Gerar **N = 200 amostras** por sistema, cada uma até `EOS` ou 512 tokens.
- Decodificar com `tokenizer.decode(...)` e salvar `.mid`.
- Tokens inválidos na decodificação: ignorar (miditok já tolera) e reportar a taxa de amostras inválidas, se houver.

### 2.6 Baseline Markov (`src/markov.py`)
- Contar n-gramas (ordem 4 é um bom ponto de partida; testar 2–6 na validação) sobre **os mesmos tokens de treino**.
- Gerar a partir de `BOS` com o mesmo comprimento máximo e mesmo N = 200.
- *Backoff* simples para ordens menores quando o contexto não foi visto.

### 2.7 Avaliação (`src/evaluate.py`)
Mesmo protocolo para **teste humano**, **Transformer** e **Markov**:

1. Ler todos os MIDIs com `muspy.read_midi`.
2. **Recortar o conjunto de teste para o mesmo nº de compassos** que as amostras geradas (ex.: primeiros 16 compassos) — métricas como *pitch range* dependem da duração. Mencionar isso no artigo (mostra cuidado metodológico).
3. Calcular por música (5 métricas, o mínimo é 3):
   - `pitch_range`
   - `n_pitch_classes_used`
   - `pitch_class_entropy`
   - `scale_consistency`
   - `groove_consistency` (ou `empty_beat_rate` / `polyphony`)
4. Comparar **distribuições** (Yang & Lerch, 2020):
   - média ± desvio de cada sistema;
   - **Overlap Area (OA)** entre a KDE do sistema e a KDE do teste (`scipy.stats.gaussian_kde`, integrar `min(p, q)`);
   - **KL divergence** entre as KDEs;
   - **violin/box plots** por métrica (uma figura com 5 subplots).
5. Extra barato (não obrigatório, mas valoriza): **perplexidade no teste** para Transformer vs. Markov.

**Tabela principal do artigo:**

| Métrica | Teste (média±dp) | Transformer (média±dp) | OA ↑ | KL ↓ | Markov (média±dp) | OA ↑ | KL ↓ |
|---|---|---|---|---|---|---|---|

### 2.8 Áudio (`src/render.py`)
- `fluidsynth -ni <soundfont.sf2> in.mid -F out.wav -r 44100` (Mac: `brew install fluid-synth`; soundfont livre, ex.: GeneralUser GS ou FluidR3_GM).
- Converter para `.mp3` para não pesar o repositório.
- Escolher **5 MIDIs do Transformer + 5 do Markov** (sorteados com seed, não “os melhores”, ou dizer no artigo que foram escolhidos a dedo) → `samples/transformer/` e `samples/markov/`.

---

## 3. Estrutura do repositório

```
README.md            # instalação, como reproduzir, contribuições de cada integrante
requirements.txt     # torch, miditok, muspy, symusic, pretty_midi, scipy, matplotlib, pandas, tqdm
src/
  config.py          # hiperparâmetros e caminhos em um só lugar
  data.py            # download, split por música, filtragem, estatísticas
  tokenize.py        # REMI + janelas
  model.py           # Transformer (PyTorch puro)
  train.py           # laço de treino + log de perdas
  sample.py          # temperatura / top-k / top-p + geração de N amostras
  markov.py          # baseline
  evaluate.py        # métricas muspy + OA/KL + gráficos
  render.py          # MIDI → WAV/MP3
results/             # CSVs de métricas, figuras, curvas de treino
samples/
  transformer/       # ≥5 .mid + .mp3
  markov/            # ≥5 .mid + .mp3
paper/               # PDF final (LaTeX fica no Overleaf, exportar no fim)
```

**Reprodução no README** deve ser literalmente uma sequência de comandos:
```
pip install -r requirements.txt
python src/data.py
python src/tokenize.py
python src/train.py
python src/sample.py --model transformer --n 200
python src/markov.py --n 200
python src/evaluate.py
python src/render.py
```

---

## 4. Divisão de tarefas (ajustar ao nº de integrantes)

| Pessoa | Responsabilidade | Também escreve no artigo |
|---|---|---|
| **A – Dados** | `data.py`, `tokenize.py`, estatísticas do dataset | Método: dataset, pré-processamento, representação |
| **B – Modelo** | `model.py`, `train.py`, `sample.py`, rodar treino no Colab | Método: arquitetura e amostragem; Experimentos: hiperparâmetros/hardware |
| **C – Baseline + Avaliação** | `markov.py`, `evaluate.py`, tabela e gráficos | Experimentos: métricas e protocolo; Resultados |
| **D – Artigo** | Overleaf com template ISMIR, Introdução, Trabalhos relacionados, Conclusão, revisão final | Integra tudo, controla o limite de 4 páginas |
| **E – Áudio/Repo/Slides** | `render.py`, README, requirements, seleção das amostras, slides da apresentação | Análise qualitativa das músicas |

Com 3 pessoas: A+E, B, C+D.

---

## 5. Cronograma

| Dia | Meta | “Pronto quando…” |
|---|---|---|
| **Qui 20/11** | Repo + `requirements.txt`; download do dataset; split; tokenização; projeto do Overleaf criado com o template ISMIR e as seções vazias | Existe `data/splits.json` e sabemos o vocabulário e o comprimento médio |
| **Sex 21/11** | **Pipeline ponta a ponta com modelo minúsculo** (1 camada, 1 época): treina → gera → decodifica → avalia. Markov pronto. `evaluate.py` roda sobre o teste humano | Um MIDI gerado toca e a tabela de métricas sai (mesmo que ruim) |
| **Sáb 22/11** | Treino completo do Transformer; ajustar ordem do Markov na validação; gerar 200 amostras de cada | Checkpoint final + 400 MIDIs gerados |
| **Dom 23/11** | Rodar avaliação final no teste; gráficos; renderizar áudios; escrever Método e Experimentos | Tabela + figuras finais no Overleaf |
| **Seg 24/11** | Escrever Resultados/Discussão/Conclusão; README com contribuições; revisão pelo checklist abaixo | Artigo completo em ≤4 páginas |
| **Ter 25/11** | **Folga** para imprevistos; revisão final; tornar o repo público; submeter no Moodle | Submetido antes das 23:59 |

A apresentação em sala vem depois: reaproveitar as figuras do artigo + tocar 2–3 áudios (1 humano, 1 Transformer, 1 Markov).

---

## 6. Esqueleto do artigo (4 páginas)

1. **Introdução** (~0,4 pág.) – motivação, tarefa (geração livre de melodias folclóricas com acompanhamento), contribuições.
2. **Trabalhos relacionados** (~0,4 pág.) – Music Transformer (Huang et al., 2019), Pop Music Transformer / REMI (Huang & Yang, 2020), cadeias de Markov em música, MidiTok (Fradet et al., 2021), avaliação objetiva (Yang & Lerch, 2020), MusPy (Dong et al., 2020).
3. **Método** (~1 pág.) – dataset + pré-processamento + estatísticas; REMI (vocabulário, comprimento médio); arquitetura; baseline; amostragem.
4. **Experimentos** (~0,5 pág.) – partições, hiperparâmetros, hardware, tempo de treino, métricas, protocolo (N amostras, recorte em 16 compassos, OA/KL).
5. **Resultados e discussão** (~1,2 pág.) – tabela principal, figura de violinos, curvas de treino/validação, análise qualitativa (repetição de motivos, cadências, coerência rítmica; mostrar um *piano roll* lado a lado: humano × Transformer × Markov). Citar os arquivos em `samples/`.
6. **Conclusão** (~0,3 pág.) – limitações (dataset pequeno, sem condicionamento, contexto curto) e trabalhos futuros (condicionamento por tonalidade, datasets maiores, teste de audição).
7. **Referências** (não contam no limite).

---

## 7. Checklist de nota total (conferir no dia 24/11)

**Tarefa / dados**
- [ ] Tarefa explicitada no artigo
- [ ] Split treino/val/teste **por música**; teste só usado na avaliação final
- [ ] Pré-processamento e estatísticas do dataset descritos

**Representação**
- [ ] Representação escolhida e **justificada pela tarefa**
- [ ] Tamanho do vocabulário e comprimento médio das sequências reportados

**Modelo**
- [ ] Transformer/LSTM/GRU em **PyTorch nativo** (sem Hugging Face, `Trainer`, `x-transformers`)
- [ ] Classe do modelo, laço de treino e amostragem (temperatura/top-k/top-p) **escritos pelo grupo**

**Baseline**
- [ ] Baseline mais simples com **mesmos dados, mesma representação e mesmo protocolo**
- [ ] Código de terceiros, se usado, citado

**Experimentos**
- [ ] ≥3 métricas objetivas (temos 5)
- [ ] ≥100 amostras por sistema (temos 200)
- [ ] Comparação de **distribuições** (OA/KL + gráficos), não só médias
- [ ] Curvas de treino e validação no artigo
- [ ] Análise qualitativa das músicas geradas

**Entregáveis**
- [ ] Repositório GitHub **público**
- [ ] `README.md` com instalação e reprodução
- [ ] `requirements.txt`
- [ ] Contribuição de cada integrante no README
- [ ] ≥5 MIDIs + áudios do modelo e ≥5 do baseline no repo, **referenciados no artigo**
- [ ] PDF no template ISMIR, ≤4 páginas de conteúdo
- [ ] Submissão no Moodle por um integrante
- [ ] Slides da apresentação

---

## 8. Riscos e planos B

| Risco | Plano B |
|---|---|
| Nottingham não baixa / MIDIs quebrados | Trocar para JSB Chorales do `music21.corpus` (offline) |
| Sem GPU / treino lento | Reduzir para `d_model=128`, 3 camadas, contexto 256; o Nottingham é pequeno o bastante para CPU/MPS |
| Modelo gera lixo ou repete a mesma nota | Conferir a máscara causal e o deslocamento entrada/alvo; baixar temperatura para 0.9; usar top-p 0.9 |
| Decodificação falha em algumas amostras | Pular essas amostras e reportar a taxa de falhas |
| Métricas muspy dão `NaN` (ex.: amostra vazia) | Filtrar amostras vazias e reportar quantas foram |
| Artigo passa de 4 páginas | Mover detalhes para o README; juntar figuras em uma só com subplots |
