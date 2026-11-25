# Pivô: pré-treino no Pop1K7 e fine-tune no EMOPIA+ com emoção e curvas

Este documento registra a mudança de rumo feita na madrugada de 25/11, depois do primeiro treino
completo no cluster. Ele continua o roadmap original (curvas de energia e tensão por compasso, com
Transformer e baseline Markov) e serve de base para as seções de método, experimentos e discussão do
artigo. Os números saem da execução do `jobs/pipeline.sh` (guia em [`pipeline.md`](pipeline.md)); a
cópia dos resultados da rodada final está em [`../resultados_artigo/`](../resultados_artigo/) e a análise
completa em [`notas_resultados.md`](notas_resultados.md).

## 1. Motivação: o que deu errado no primeiro treino

O primeiro treino seguiu o roadmap à risca: EMOPIA 1.0, REMI, tokens `Energy`/`Tension` depois de
cada `Bar`, Transformer grande (19,6M) e pequeno (3,4M) do zero. Os números de treino pareciam
razoáveis (perplexidade de validação de 10,2 no pequeno, contra 32,7 do Markov), mas a escuta foi
ruim: um integrante músico avaliou que as amostras não seguiam uma linha de raciocínio musical. Não
havia tonalidade estável, escala coerente nem variação rítmica com sentido. Baixar a temperatura
(0,9) melhorava pouco.

A investigação encontrou a causa nos dados, não no modelo. Os arquivos de `midis/` do EMOPIA 1.0
são a transcrição automática direta do áudio (GiantMIDI-Piano). Todos declaram 120 BPM e 4/4, mas
esse andamento é um valor padrão: as notas não estão alinhadas a nenhum pulso. Medimos a posição de
cada início de nota dentro do beat da grade de 120 BPM:

| Conjunto | Inícios a menos de 5% de uma semicolcheia | Esperado se fosse ao acaso |
|---|---|---|
| EMOPIA 1.0 (`midis/`) | 9,5% | 10% |
| EMOPIA+ (`midis/`) | 91,3% | 10% |
| Pop1K7 (`midi_synchronized/`) | 31,6% (beats alinhados, timing expressivo dentro do beat) | 10% |

Consequências para o primeiro treino:

- o "compasso" que o modelo via era uma janela de 2 segundos sem relação com o compasso musical;
  as curvas de energia e tensão eram medidas nessas janelas;
- os tokens `Position` do REMI eram, na prática, ruído: a mesma figura rítmica aparecia em posições
  diferentes a cada repetição, então não havia regularidade métrica para aprender;
- com só 876 clipes de treino (cerca de 0,85M de tokens originais), o modelo grande decorou o treino
  em 4 épocas (perda de treino 1,60 contra 3,20 na validação na época 9), e o pequeno em 7.

Os próprios autores do EMOPIA não geram a partir desses MIDIs crus: eles sincronizam a transcrição
com o beat detectado no áudio e fazem pré-treino num conjunto maior (Pop1K7) antes de ajustar no
EMOPIA [Hung et al., 2021].

## 2. Escolha dos dados

| Conjunto | O que é | Decisão |
|---|---|---|
| **EMOPIA+** [Huang et al., 2024] | Os mesmos clipes do EMOPIA (1071), mesmos nomes e mesma divisão, quantizados no beat, separados em faixas `Melody`, `Texture`, `Bass` e `Chord` | **Substitui o EMOPIA 1.0.** Mesmo desenho experimental, agora com compassos de verdade |
| **Pop1K7** [Hsiao et al., 2021] | 1747 covers de piano pop (cerca de 108 h, 12 vezes o EMOPIA), transcritos e sincronizados com o beat pelo mesmo laboratório | **Pré-treino.** Mesmo gênero, mesmo instrumento, mesmo pipeline de transcrição |
| MAESTRO, GiantMIDI-Piano | Piano clássico tocado ao vivo, sem grade de beat | Descartados: repetiriam o problema rítmico, e o gênero é outro |
| POP909 | 909 arranjos pop em várias faixas | Descartado: menor que o Pop1K7 e com formato diferente (melodia, ponte e piano separados) |

Os dois conjuntos escolhidos têm licença CC-BY 4.0 e são baixados do Zenodo por
`scripts/download_data.sh`.

Tamanho depois do pré-processamento (`src.data.build_dataset`):

| Partição | Peças | Tokens (com transposição no treino) | Tokens por peça (média) |
|---|---|---|---|
| EMOPIA+ treino | 869 clipes × até 12 transposições | 9,9M | 975 |
| EMOPIA+ validação / teste | 114 / 88 clipes | 116k / 83k | 1021 / 943 |
| Pop1K7 pré-treino | 1657 músicas × até 7 transposições | 57,4M (cerca de 8,2M originais) | 4957 |
| Pop1K7 validação | 87 músicas | 453k | 5208 |

Detalhes medidos que entram no pré-processamento:

- EMOPIA+: tpq 480, sempre 4/4, mediana de 18 compassos por clipe. A faixa `Chord` é uma anotação
  de acordes, não parte da execução: juntamos `Melody`, `Texture` e `Bass` numa única faixa de piano
  (17 clipes não têm `Texture`).
- Pop1K7: uma faixa, tpq 480, 4/4, mediana de 416 beats (cerca de 104 compassos) por música.
- Grade rítmica: no EMOPIA+, 85,5% dos inícios caem na grade de 1/4 de beat e 92,5% na de 1/12,
  porque há tercinas. A grade de 1/8 de beat do roadmap não acrescenta nada à de 1/4 e não captura
  tercinas, então o REMI passa a usar **12 posições por beat** no primeiro compasso de duração.

## 3. Desenho em duas fases

1. **Pré-treino no Pop1K7.** O modelo aprende linguagem musical de piano pop: tonalidade,
   progressões, condução de vozes, métrica. As sequências já levam os tokens de energia e tensão
   medidos com a mesma função, então o pré-treino também ensina o modelo a seguir as curvas.
2. **Fine-tune no EMOPIA+.** Os pesos do pré-treino são o ponto de partida; o otimizador e o
   scheduler recomeçam com learning rate menor (1e-4). Nesta fase entra a emoção.

O desenho ataca os dois problemas do primeiro treino: o EMOPIA+ corrige o ritmo e o Pop1K7 dá
volume de dados (o overfitting do modelo grande vinha da falta de dados, não de excesso de
capacidade).

## 4. Condicionamento: emoção global e curvas por compasso

A sequência de cada música passa a ser:

```
BOS Emotion_Qx Bar Energy_e Tension_t <notas do compasso 1> Bar Energy_e Tension_t <notas> ... EOS
```

- `Emotion_Q1..Q4` é o quadrante de Russell do clipe no EMOPIA (Q1 alegre/excitado, Q2 tenso/
  agitado, Q3 triste/calmo, Q4 sereno). O Pop1K7 não tem rótulo de emoção e recebe `Emotion_None`.
  Com isso, o mesmo vocabulário serve às duas fases e o modelo aprende no fine-tune o que cada
  quadrante muda em relação ao "neutro".
- As curvas de energia e tensão seguem o contrato 3 do roadmap, com uma diferença importante: agora
  cada `Bar` é um compasso musical de verdade.
- O token de emoção nunca sai do contexto: toda janela de treino que não começa no início da música
  recebe `BOS Emotion_Qx` na frente, e a geração faz o mesmo quando o prefixo passa do contexto.
- Na geração, `Emotion` e as curvas são forçados (nunca amostrados), como os controles já eram.

O vocabulário passa de 289 para 326 tokens: 5 de emoção e as posições e durações da grade de 1/12
(48 posições por compasso, antes 32).

Com os limites recalculados no treino do EMOPIA+ (energia 0,360 e 0,497; tensão 0,067 e 0,145), os
níveis ficam em 33/33/33, a correlação de Spearman entre energia e tensão é de 0,15, e a energia
média separa os quadrantes de excitação: Q1 0,51 e Q2 0,57, contra Q3 0,35 e Q4 0,35. No Pop1K7, com
os mesmos limites, a energia fica em 37/49/15% e a tensão em 28/38/34%.

## 5. Vazamento entre Pop1K7 e EMOPIA

Os dois conjuntos vêm de covers de piano do YouTube coletados pelo mesmo laboratório, então uma
mesma música pode estar nos dois. Uma música do teste do EMOPIA+ que aparece no pré-treino
inflaria as métricas. Antes do pré-treino, `src/data/leakage.py` compara impressões digitais
(n-gramas de 8 alturas consecutivas) de cada música do Pop1K7 com cada clipe de validação e de
teste do EMOPIA+ e **remove do pré-treino** toda música que contém mais de 30% dos n-gramas de
algum desses clipes.

Para calibrar o limiar, o mesmo cálculo entre clipes de validação/teste e músicas do treino do
EMOPIA+ (que são outras músicas, por construção da divisão) dá mediana de 0,015 e p95 de 0,091.

Resultado (`configs/pop1k7_leakage.json`): **3 músicas do Pop1K7 removidas**. Duas delas
(`src_002/279` e `src_002/367`) contêm de 51% a 66% dos n-gramas de todos os clipes de duas músicas
de validação/teste do EMOPIA+, ou seja, são o mesmo cover; a terceira passa um pouco do limiar
(0,32). Sobram 1744 músicas: 1657 no pré-treino e 87 na validação do pré-treino.

O cálculo de referência mostrou também um caso dentro do próprio EMOPIA: um clipe de validação/teste
com contenção de 0,85 numa música do treino com outro id do YouTube (provavelmente dois covers da
mesma composição). É uma característica da divisão oficial, que mantemos para comparar com a
literatura, e fica registrada como limitação.

## 6. Baseline com os mesmos dados

Para a comparação continuar justa, o Markov é ajustado nos mesmos dados que o Transformer viu:
Pop1K7 (sem transposição, para caber na memória) mais o treino do EMOPIA+ com transposição, num
total de 18,2M de tokens. A ordem n e o desconto D continuam escolhidos na validação do EMOPIA+.
Resultado: n = 4, D = 0,9, perplexidade de validação de 25,7 (no EMOPIA 1.0 era 32,7, com outro
vocabulário; os números não são diretamente comparáveis).

## 7. Experimentos

| Execução | Dados | Papel no artigo |
|---|---|---|
| `pretrain_large`, `pretrain_small` | Pop1K7 | Fase 1 |
| `finetune_large`, `finetune_small` | EMOPIA+, a partir do pré-treino | Sistema principal |
| `scratch_large`, `scratch_small` | EMOPIA+, do zero | Ablação: quanto o pré-treino ajuda |
| Markov | Pop1K7 + EMOPIA+ | Baseline |

Avaliação, sempre com o teste do EMOPIA+:

- métricas do muspy com OA e KL [Yang e Lerch, 2020], como no roadmap;
- fidelidade às curvas pedida contra embaralhada, como no roadmap;
- **novo: emoção por quadrante pedido.** Sem um classificador de emoção, usamos dois indicadores
  medidos nas músicas geradas: energia média (ligada à excitação: Q1 e Q2 devem ter mais que Q3 e
  Q4) e fração de músicas em modo maior pela tonalidade estimada (ligada à valência: Q1 e Q4 devem
  ter mais que Q2 e Q3). A referência humana mostra o tamanho esperado da separação.

Resultados (de `resultados_artigo/treino/summary.md`; detalhes em [`notas_resultados.md`](notas_resultados.md)):

| | Perplexidade de validação (EMOPIA+) |
|---|---|
| Markov | 25,7 |
| `scratch_small` / `scratch_large` | 6,72 / 8,39 |
| `finetune_small` / `finetune_large` | 5,02 / 4,76 |

## 8. O que muda em cada contrato do roadmap

| Contrato | Antes | Agora |
|---|---|---|
| 1. Estrutura | `src/data/emopia.py` | mais `src/data/pop1k7.py` e `src/data/leakage.py` |
| 2. Tokenizador | REMI, 1/8 de beat, 289 tokens | REMI, 1/12 de beat, tokens `Emotion_*` |
| 3. Tokens de controle | `Bar Energy Tension` | `BOS Emotion` no início, depois `Bar Energy Tension` |
| 4. Medição | `measure_bar` em janelas de 2 s | mesma função, em compassos de verdade |
| 5. Split | reprodução do notebook do EMOPIA 1.0 | CSVs oficiais do EMOPIA+ (conferimos: as músicas de cada partição são exatamente as da nossa reprodução); Pop1K7 com 5% das músicas para validação do pré-treino |
| 6. Interface do modelo | `next_token_logits` | igual |
| 7. Curvas de avaliação | curvas reais do teste | curvas e quadrante reais do teste |

Os tercis dos níveis de energia e tensão são recalculados no treino do EMOPIA+ e valem também para o
Pop1K7.

## 9. Limitações

- A tonalidade é estimada uma vez por música. Nas músicas longas do Pop1K7, com modulações, a
  tensão medida por notas fora da escala fica menos precisa.
- Só o EMOPIA tem rótulo de emoção, e são 876 clipes de treino: o modelo aprende o efeito de cada
  quadrante com pouco dado.
- Os indicadores de emoção (energia e modo maior) são aproximações; não há avaliação perceptual. Nos
  próprios clipes humanos do teste, o modo maior separa bem Q2 (16%) de Q1 e Q4 (88% e 78%), mas Q3
  (triste) ainda tem 48% de músicas em modo maior.
- O REMI não tem tokens de andamento: as músicas são renderizadas num andamento fixo, igual para as
  humanas e as geradas.
- O Pop1K7 é transcrição automática; erros de transcrição entram no pré-treino.

## Referências

- Hung, H.-T. et al. EMOPIA: A Multi-Modal Pop Piano Dataset for Emotion Recognition and
  Emotion-based Music Generation. ISMIR, 2021. Dados: https://zenodo.org/records/5090631
- Huang, J., Chen, K., Yang, Y.-H. Emotion-driven Piano Music Generation via Two-stage
  Disentanglement and Functional Representation. ISMIR, 2024. EMOPIA+:
  https://zenodo.org/records/13122742
- Hsiao, W.-Y. et al. Compound Word Transformer: Learning to Compose Full-Song Music over Dynamic
  Directed Hypergraphs. AAAI, 2021. Pop1K7: https://zenodo.org/records/13167761
- Huang, Y.-S., Yang, Y.-H. Pop Music Transformer: Beat-based Modeling and Generation of Expressive
  Pop Piano Compositions (REMI). ACM Multimedia, 2020.
- Kong, Q. et al. GiantMIDI-Piano: A large-scale MIDI dataset for classical piano music. TISMIR, 2022.
- Yang, L.-C., Lerch, A. On the evaluation of generative models in music. Neural Computing and
  Applications, 2020.
