# Músicas geradas (MIDI), rodada do cluster de 25/11

- `teste/`: as 100 amostras de cada sistema usadas nas tabelas de `../teste/`, todas com
  t = 0,9 e top-p = 1,0 (Markov incluído). `human/` são os 88 clipes do teste do EMOPIA+.
- `validacao_grade/`: as 9 combinações de temperatura × top-p do finetune_large na validação
  (as de `../validacao/`), mais os 114 clipes humanos da validação.

Cada pasta tem um `manifest.json`. Para cada arquivo, ele traz:

| Campo | O que é |
|---|---|
| `curve_clip` | clipe humano de onde veio a curva pedida; o prefixo `Q1`..`Q4` é o quadrante |
| `emotion` | quadrante pedido, de 1 a 4 |
| `requested_levels` | pares [energia, tensão] por compasso, com níveis 0, 1 e 2 |
| `finished` | se a música terminou a curva |

O mesmo número de arquivo em sistemas diferentes do teste usa a mesma curva e a mesma emoção:
`teste/finetune_large/000.mid` e `teste/markov/000.mid` receberam o mesmo pedido, o que dá uma
comparação lado a lado.

Para ouvir: qualquer player de MIDI (MuseScore, VLC com soundfont) ou
`fluidsynth -ni <soundfont.sf2> arquivo.mid -F saida.wav`.
