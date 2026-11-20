# Roadmap TP2 — Curvas de Energia e Tensão

Nov 23, 2026 · @Natan

## Visão geral

São cerca de 55 horas até a entrega (qua 25/11, 23:59). O plano roda as cinco frentes em paralelo e depende de um pipeline mínimo funcionando de ponta a ponta até **terça 14h**, para que o treino completo rode no cluster na noite de terça. A meta é submeter na quarta às 22h, com 2h de folga.

Premissas: o grupo é só de graduação (artigo de até 4 páginas, sem experimento extra obrigatório), vai treinar no cluster do DCC e ainda não implementou nada.

| Pessoa | Frente | Entrega principal |
| --- | --- | --- |
| Pedro | Dados e representação, depois edição do artigo | Split por música, tokenização REMI, estatísticas do dataset |
| Julia M. | Atributos de controle | Energia e tensão por compasso, quantização, tokens de controle |
| Natanael | Transformer | Modelo, laço de treino, amostragem, treino no cluster |
| Leonardo | Baseline, geração e repositório | Markov de ordem n, rotina de geração condicionada comum, áudios, README |
| Júlia B. | Avaliação | Métricas do muspy, OA/KL, fidelidade ao controle, experimento com curvas embaralhadas |

A divisão é uma sugestão equilibrada por carga de trabalho. Troquem os nomes à vontade, mas mantenham uma pessoa por frente.

## Contratos a congelar hoje (seg, 23h)

Com estes sete pontos definidos hoje à noite, as cinco frentes trabalham ao mesmo tempo sem esperar umas pelas outras. Cada um tem um responsável e fica registrado no README.

1. **Estrutura do repositório** (Leonardo): `src/data`, `src/controls`, `src/model`, `src/baseline`, `src/generate`, `src/eval`, `configs/`, `outputs/`, com seed fixa global.
2. **Tokenizador** (Pedro): REMI do miditok com configuração salva em `tokenizer.json` e não mexida depois disso. Os tokens de controle já entram reservados no vocabulário.
3. **Tokens de controle** (Julia M.): `Energy_0..K-1` e `Tension_0..K-1`, sugestão K = 3. Ordem em cada compasso: `Bar`, `Energy`, `Tension` e depois as notas. Os limites dos níveis são tercis calculados **só no treino** e salvos em JSON.
4. **Função única de medição** (Julia M.): `measure_bar(notas, tonalidade) -> (energia, tensão)`. É a mesma função que rotula o treino e que mede depois as músicas geradas. Ter duas versões dela invalidaria a métrica de fidelidade.
5. **Split** (Pedro): usar a partição oficial do EMOPIA, se houver uma por música. Se não houver, agrupar os trechos pelo ID da música e dividir 80/10/10 com seed fixa. O resultado fica salvo em `splits.json`, e o teste só é usado na quarta.
6. **Interface do modelo** (Natanael e Leonardo): Transformer e Markov expõem o mesmo método, `next_token_logits(prefixo) -> logits`. Assim a amostragem e o forçamento dos controles são o mesmo código para os dois.
7. **Curvas de avaliação** (Júlia B.): as curvas reais medidas nas músicas de teste, gerando 100 ou mais amostras por sistema. Para o experimento embaralhado, as curvas são permutadas entre músicas, com seed fixa.

## Cronograma

&#91;embedded content: cronograma · 5 fases, 4 marcos e a submissão\]

O G2 é o único marco que não pode escorregar, porque a noite de terça é a única janela longa de treino no cluster.

## Tarefas por pessoa

Cada tarefa tem o horário em que precisa estar pronta. As marcadas com G2 são pré-requisito do teste de ponta a ponta de terça às 14h.

### Pedro: dados e representação

- [ ] **Seg 21h**: baixar o EMOPIA no cluster, montar o split por música e salvar `splits.json`
- [ ] **Seg 23h**: congelar `tokenizer.json` (REMI, com os tokens de controle reservados)
- [ ] **Ter 10h** (G2): tokenizar tudo, aplicar transposição como data augmentation só no treino, recortar em janelas e salvar o dataset em PyTorch
- [ ] **Ter 12h**: estatísticas do dataset (trechos e músicas por partição, tamanho do vocabulário, comprimento médio das sequências)
- [ ] **Ter 14h em diante**: assumir a edição do artigo, com o template ISMIR no Overleaf e as seções da tabela abaixo

### Julia M.: atributos de controle

- [ ] **Seg 23h**: especificar por escrito as fórmulas de energia e tensão
- [ ] **Ter 10h** (G2): implementar `measure_bar`
  - Energia: densidade de notas por compasso e velocity média, normalizadas
  - Tensão: proporção de notas fora da escala, com a tonalidade estimada por trecho (music21), mais a taxa de intervalos dissonantes entre notas simultâneas
- [ ] **Ter 10h** (G2): quantizar com os tercis do treino e inserir os tokens depois de cada `Bar`, junto com o Pedro
- [ ] **Ter 13h**: verificações de sanidade (balanceamento dos níveis, correlação entre energia e tensão, gráfico das curvas de algumas músicas)
- [ ] **Ter 18h**: gerar as curvas embaralhadas para o experimento, junto com a Júlia B.
- [ ] **Qua**: análise qualitativa ouvindo as amostras e montagem dos slides da apresentação

### Natanael: Transformer

- [ ] **Seg 23h**: esboçar a classe do modelo, com `nn.Embedding`, embedding posicional aprendido, `nn.TransformerEncoderLayer` com máscara causal e cabeça linear
- [ ] **Ter 12h** (G2): laço de treino com AdamW, warmup e decaimento cosseno, gradient clipping, checkpoints e log das curvas de treino e validação
- [ ] **Ter 12h** (G2): amostragem própria com temperatura, top-k e top-p, ligada à rotina de geração do Leonardo
- [ ] **Ter 15h**: disparar o treino completo no cluster. Sugestão de partida: 6 camadas, d\_model 512, 8 cabeças, contexto de 1024 tokens, dropout 0,1. Disparar junto um modelo pequeno de segurança
- [ ] **Qua 08h**: escolher o checkpoint pela loss de validação e exportar as curvas de treino
- [ ] **Qua 10h**: ajustar temperatura, top-k e top-p **na validação**, nunca no teste

### Leonardo: baseline, geração e repositório

- [ ] **Seg 20h**: criar o repositório público no GitHub com a estrutura de pastas, o `requirements.txt` e o script de job do cluster, e dar acesso a todos
- [ ] **Ter 10h** (G2): cadeia de Markov de ordem n sobre os tokens, com backoff para contextos não vistos e n escolhido pela verossimilhança na validação
- [ ] **Ter 12h** (G2): rotina de geração comum `generate(modelo, curva, amostragem)`. A cada `Bar` gerado, ela força os tokens de energia e tensão da curva, para no número de compassos pedido e decodifica para MIDI
- [ ] **Qua 12h**: gerar 100 ou mais amostras por sistema, com as curvas reais e com as embaralhadas
- [ ] **Qua 14h**: renderizar áudio com FluidSynth e um soundfont livre, e subir 5 MIDIs + áudios de cada sistema para o repositório

### Júlia B.: avaliação

- [ ] **Ter 12h** (G2): script com as métricas do muspy (amplitude de alturas, entropia de classes de altura, consistência de escala, consistência rítmica, polifonia)
- [ ] **Ter 16h**: OA e KL no estilo de Yang e Lerch (2020), com distâncias intra e inter-conjunto e KDE. Para checar a sanidade, comparar validação com teste: a sobreposição tem que sair alta
- [ ] **Ter 18h**: métricas de fidelidade compasso a compasso, separadas para energia e tensão: acurácia exata, acurácia com tolerância de ±1 nível e correlação de Spearman
- [ ] **Ter 18h**: protocolo do experimento embaralhado. A fidelidade à curva pedida tem que ser maior que a fidelidade a uma curva não usada; se for igual, o modelo está ignorando o controle
- [ ] **Qua 16h**: congelar tabelas e figuras (distribuições das métricas, curva pedida contra curva obtida)

## Artigo ISMIR

| Seção | Responsável | Rascunho até |
| --- | --- | --- |
| Introdução e trabalhos relacionados | Pedro | Ter 20h |
| Método: dataset, pré-processamento e REMI | Pedro | Ter 20h |
| Método: energia, tensão e tokens de controle | Julia M. | Ter 20h |
| Método: Transformer e amostragem | Natanael | Ter 22h |
| Método: baseline Markov | Leonardo | Ter 22h |
| Experimentos: hiperparâmetros e hardware | Natanael | Qua 12h |
| Experimentos: métricas e protocolo | Júlia B. | Ter 22h |
| Resultados e discussão | Júlia B., com análise qualitativa da Julia M. | Qua 18h |
| Conclusão, limitações e trabalhos futuros | Pedro | Qua 19h |
| Revisão final em conjunto | Todos | Qua 21h |

Nos trabalhos relacionados, o mínimo é citar Music Transformer, Pop Music Transformer (REMI), o artigo do EMOPIA, um trabalho de geração com controle por atributos e Yang e Lerch (2020).

## Riscos e checklist de entrega

O maior risco é o G2 (pipeline de ponta a ponta às 14h de terça) atrasar e não sobrar uma noite inteira para o treino. Se ele não fechar até terça 16h, o escopo congela no modelo pequeno.

| Risco | Plano B |
| --- | --- |
| Fila do cluster cheia ou treino que não converge | Modelo pequeno de segurança (4 camadas, d\_model 256, contexto de 512 tokens) disparado junto com o grande na terça às 15h |
| Modelo que ignora os tokens de controle | É um resultado válido: o experimento embaralhado mostra isso, e o artigo discute. Antes, conferir o balanceamento dos níveis |
| Tonalidade mal estimada em trechos curtos | Estimar uma tonalidade por trecho inteiro e documentar a limitação |
| Markov esparso em ordem alta | Backoff e n pequeno, escolhido na validação |
| Vazamento de dados | Split por música salvo na segunda, e o teste só é aberto na quarta |
| Artigo passando de 4 páginas | Cortar texto e mandar figuras secundárias para o README, sem reduzir a fonte |

Checklist da entrega (Moodle, por um integrante, até qua 23:59):

- [ ] Repositório público com o código de pré-processamento, treino, geração e avaliação
- [ ] `requirements.txt`
- [ ] 5 MIDIs do Transformer e 5 do Markov, com áudios, no repositório e citados no artigo
- [ ] PDF do artigo no template ISMIR, com até 4 páginas de conteúdo
- [ ] Link do repositório e PDF enviados no Moodle

Em aberto: a data da apresentação em sala. Os slides ficam com a Julia M. e podem ser feitos depois da submissão, se a apresentação for mais tarde.
