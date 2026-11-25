# Notas dos resultados do cluster (rodada de 25/11)

Fonte: `outputs_cluster/` (rsync do cluster). Todas as execuções rodaram numa NVIDIA L40S com bf16.
O teste foi gerado com a amostragem escolhida na validação, **t = 0,9 e top-p = 1,0**, com 100 amostras
por sistema e seed 42.

> **Atenção:**
> - Em `outputs_cluster/` ainda há resultados do EMOPIA 1.0, de antes do pivô: `runs/large`, `runs/small`,
>   `samples/large*`, `samples/small*` e `figures/training_curves.*`. Eles usam outro tokenizador e outros dados.
> - Não misture esses resultados com as tabelas abaixo. A val loss deles (2,59 e 2,32) não é comparável
>   com a de agora.
> - Eles só servem como "antes" no áudio ou na motivação.

## 1. Treino (`runs/summary.md`)

| Execução | Parâmetros | Melhor época / total | val loss | Perplexidade | s/época |
|---|---|---|---|---|---|
| pretrain_large | 19,6M | 4 / 7 | 1,537 | 4,65 | 275 |
| pretrain_small | 3,4M | 18 / 20 | 1,574 | 4,83 | 79 |
| **finetune_large** | 19,6M | **1** / 4 | **1,561** | **4,76** | 41 |
| finetune_small | 3,4M | 2 / 5 | 1,614 | 5,02 | 13 |
| scratch_large | 19,6M | 5 / 10 | 2,127 | 8,39 | 41 |
| scratch_small | 3,4M | 5 / 10 | 1,906 | 6,72 | 13 |

- **Ganho do pré-treino** (val loss do zero − fine-tune):
  - large: **+0,57 nats**, e a perplexidade cai de 8,39 para 4,76;
  - small: **+0,29 nats**.
- **Do zero, o large decora o EMOPIA+:**
  - perda de treino 0,94 contra 2,82 de validação na época 10;
  - por isso fica pior que o small do zero (2,13 contra 1,91), o mesmo comportamento do EMOPIA 1.0;
  - com o pré-treino, a ordem se inverte e o large passa a ser o melhor.
- **O fine-tune do large atinge o melhor resultado já na época 1** (489 passos) e depois só sobreajusta: 1,56, 1,60, 1,67, 1,76.
  - Quase todo o ganho vem do pré-treino, e o fine-tune só precisa "encostar" no EMOPIA+.
  - Para o artigo: com validação mais frequente, ou com lr menor, talvez desse para ganhar um pouco mais. Fica como limitação ou trabalho futuro, não é necessário.
- **O pretrain_large sobreajusta a partir da época 4:**
  - a validação sobe de 1,537 para 1,590, enquanto a perda de treino cai para 1,32;
  - os 57M de tokens são, na verdade, cerca de 8M únicos (1657 músicas × 7 transposições), e a transposição não traz informação nova.
- **O pretrain_small ainda melhorava devagar na época 20,** já no lr mínimo: 1,5741 na época 18, contra 1,5747 na 20. Para todos os efeitos, convergiu.
- **Validação abaixo do treino no small:**
  - a diferença cai a cada época: 0,66, 0,15, 0,10, 0,08 … 0,02 na época 20;
  - é o padrão de média da época com dropout, e não de vazamento, que manteria ou abriria a diferença;
  - isso fecha a investigação de ontem.

Custo total aproximado:

| Etapa | Tempo |
|---|---|
| Pré-treino large | 7 × 275 s ≈ 32 min |
| Pré-treino small | 20 × 79 s ≈ 26 min |
| Fine-tune / do zero | 41 s/época (large) e 13 s/época (small) |
| Geração de 100 músicas no teste | 57 s (finetune_large), 18 s (finetune_small), 8 s (Markov) |

## 2. Teste: semelhança com os humanos (`eval/test/tables.md`)

OA média das 7 métricas (quanto maior, melhor). O teto é o `human:val`, que compara a validação humana com o teste:

| Sistema | OA média | n.º de classes de altura (humano: 8,43) | scale_consistency (humano: 0,961), OA | pitch_class_entropy, OA |
|---|---|---|---|---|
| human:val (teto) | 0,940 | 8,45 | 0,896 | 0,978 |
| finetune_small | **0,929** | 8,87 | 0,939 | 0,820 |
| finetune_large | **0,927** | 8,65 | 0,928 | 0,826 |
| scratch_small | 0,898 | 10,18 | 0,834 | 0,859 |
| scratch_large | 0,896 | 10,18 | 0,861 | 0,816 |
| markov | 0,622 | 12,00 ± 0 | 0,116 | 0,165 |

- **O pré-treino corrige a tonalidade.** Era exatamente a queixa de ouvido:
  - do zero, o modelo usa ~10,2 classes de altura por clipe, contra 8,4 nos humanos, ou seja, cromatismo errante;
  - com pré-treino, usa 8,7, como os humanos;
  - a OA da `scale_consistency` também sobe, de 0,86 para 0,93.
- **Os fine-tunes ficam a ~0,01 do teto humano** na OA média.
- **O Markov não tem centro tonal:**
  - usa as 12 classes em todas as músicas;
  - a entropia de classes de altura é de 3,54 bits, quase o máximo de log₂12 = 3,58;
  - motivo: o 4-grama vê só os últimos 3 tokens, isto é, cerca de 1 nota (`Pitch Velocity Duration`), e por isso não guarda tonalidade;
  - o ritmo, por outro lado, fica razoável (polyphony e empty_beat com OA ~0,9).

## 3. Teste: fidelidade às curvas (compasso a compasso)

| Sistema | Energia exata | Energia ρ | Tensão exata | Tensão ρ |
|---|---|---|---|---|
| finetune_large | 0,749 | 0,811 | **0,542** | **0,483** |
| finetune_small | **0,788** | **0,836** | 0,527 | 0,419 |
| scratch_large | 0,710 | 0,780 | 0,445 | 0,200 |
| scratch_small | 0,696 | 0,755 | 0,413 | 0,203 |
| markov | 0,328 | −0,04 | 0,332 | 0,03 |
| (curva embaralhada, qualquer sistema) | ~0,33 | ~0 | ~0,35 | ~0 |

- **Os controles funcionam:** com a curva pedida, a fidelidade é muito maior do que com a embaralhada. A embaralhada fica no nível do acaso, o que mostra que o modelo segue a curva, e não uma média.
- **O pré-treino mais que dobra a aderência à tensão** (ρ de 0,20 para 0,48). Isso faz sentido: tensão é harmonia, e o pré-treino é o que dá harmonia ao modelo.
- **Energia é fácil, tensão é difícil.** A energia sai bem até do zero (0,78).
- **O Markov ignora as curvas.** O token de controle fica fora do contexto curto dele. É a comparação esperada.

## 4. Teste: emoção por quadrante pedido (energia média / fração em modo maior)

| Sistema | Gap de excitação | Gap de valência |
|---|---|---|
| human | +0,185 | **+0,51** |
| finetune_large | +0,183 | +0,11 |
| finetune_small | +0,188 | +0,01 |
| scratch_large | +0,211 | +0,12 |
| scratch_small | +0,178 | +0,10 |
| markov | −0,013 | +0,05 |

- **Excitação:** todos os Transformers reproduzem o gap humano (~+0,18). **Mas há confusão de fatores:**
  - a curva de energia pedida vem de um clipe do próprio quadrante;
  - então a excitação pode estar vindo da curva, e não do token de emoção;
  - o Markov, que ignora as curvas, fica em ~0.
  - Escrever isso com honestidade no artigo.
- **Valência:** este é o teste real do token, e o efeito é fraco:
  - nos humanos, Q2 (tenso e triste) tem 16% de músicas em maior; no finetune_large, 37%;
  - o gap é de +0,11 contra +0,51 dos humanos, e de +0,27 na validação humana;
  - o pré-treino não ajuda aqui: +0,11 com pré-treino contra +0,12 do zero.
  - **Limitação:** com ~870 clipes rotulados, o modelo aprende pouco a associar quadrante a modo. O pré-treino é todo `Emotion_None`.
- **Ideia barata (só geração, sem treino):** se der tempo, gerar com a **mesma curva** e o token de emoção trocado (por exemplo, curva de Q1 com token Q1 contra token Q3) e medir o modo. Isso isolaria o efeito do token. Não está implementado.

## 5. Ajuste da amostragem na validação (`eval/val/`)

| Configuração | OA média | Faixa de alturas (humano: 50,1) | Gap de valência |
|---|---|---|---|
| **t0.9 p1.0 (escolhida)** | **0,936** | 47,0 | +0,00 |
| t1.0 p1.0 | 0,935 | 56,2 | −0,00 |
| t1.0 p0.95 | 0,925 | 45,1 | −0,10 |
| t0.9 p0.95 | 0,871 | 42,1 | +0,01 |
| t0.8 p0.9 | 0,776 | 32,6 | −0,07 |

- **Achado:** o top-p < 1 piora o resultado.
  - O núcleo corta as notas menos prováveis e a música empobrece: com t0.8 p0.9, a faixa de alturas é de 32 contra 50 nos humanos, e a OA da entropia de classes de altura cai para 0,50.
  - Isso indica que o modelo está bem calibrado e que não é preciso truncar a distribuição.
- **A fidelidade sobe com a temperatura** (tensão ρ de 0,26 em t0.8 p0.9 para 0,49 em t1.0 p1.0). A amostragem conservadora "trava" num padrão e ignora a curva.
- O critério de escolha está em `pipeline.md`: maior OA média entre as configurações com ganho positivo de fidelidade nas duas curvas. Todas as nove têm esse ganho.

## 6. Large × small

- **Na val loss, o large é melhor:** 1,561 contra 1,614.
- **No teste, empatam:** OA de 0,927 contra 0,929. A energia é melhor no small, e a tensão e a valência, no large.
- **Custo:** o small tem 5,8× menos parâmetros e é 3× mais rápido.
- **Sugestão para o artigo:** o large é o sistema principal, já que foi escolhido pela validação, e o small fica como evidência de que o ganho vem dos dados (pré-treino), não do tamanho.

## 7. Figuras

| Figura | Comentário |
|---|---|
| `figures/pretrain_curves.pdf` | boa como está |
| `figures/emopia_curves.pdf` | mostra bem a ablação (o do zero diverge, o fine-tune parte de ~1,6); **a legenda cobre as curvas do scratch**, e dá para mover a legenda para fora ou reduzi-la antes de pôr no artigo |
| `eval/test/curves.pdf` | um exemplo por sistema; o laranja tem um pequeno deslocamento vertical para ficar visível; o painel `human:val` usa outro clipe (7 compassos), então não compare o painel humano visualmente com os outros |
| `figures/training_curves.*` | do EMOPIA 1.0, não usar |

## 8. Pendências práticas

- **`outputs_cluster/` não está no `.gitignore`** e tem `best.pt`/`last.pt`, que são grandes. Cuidado com `git add .`.
- **O `render` lê `outputs/samples/`:** para renderizar os áudios do cluster, troque as pastas antes de rodar:
  ```bash
  mv outputs outputs_local && mv outputs_cluster outputs
  uv run python -m src.generate.render --soundfont /caminho/FluidR3_GM.sf2 finetune_large markov human
  ```
- **Preencher a seção de resultados** de `docs/pivo_pretreino.md` com as tabelas 1 a 4 acima.
