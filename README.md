# Detecção de fraudes em cartão de crédito

Projeto de estudo com dados reais de transações de cartão de crédito. O objetivo não é apenas obter 99% de acurácia, mas construir um classificador que encontre fraudes (recall), com número controlado de falsos alertas (precisão) e decisões explicáveis (SHAP).

## 1. O problema e o desbalanceamento

A base possui 284.807 transações, das quais apenas 492 são fraudes:

| Classe | Quantidade | Proporção |
| :--- | ---: | ---: |
| 0 — normal | 284.315 | 99,8273% |
| 1 — fraude | 492 | 0,1727% |

Um modelo que sempre respondesse *normal* teria **99,8273% de acurácia** e, ainda assim, **recall = 0** para fraude: não detectaria nenhuma das 492 fraudes. Por isso a avaliação sai da acurácia e vai para a classe minoritária:

- **Recall (fraude)**: das fraudes reais, quantas o modelo encontrou.
- **Precisão (fraude)**: das transações marcadas como fraude, quantas eram fraude de fato.
- **F1 (fraude)**: média harmônica entre precisão e recall, útil quando os dois custos importam.
- **ROC-AUC e Average Precision (AP)**: qualidade das probabilidades em diferentes limiares; a curva Precision-Recall é a mais informativa quando a classe positiva é rara.

## 2. Base de dados

- Fonte: <https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv>
- Carregamento: `pd.read_csv(url)` direto no notebook — o CSV **não** é versionado (passa do limite de 100 MB do GitHub).
- Colunas: `Time`, `V1`–`V28` (componentes PCA anônimas), `Amount` e `Class` (0 = normal, 1 = fraude).

## 3. Estrutura do repositório

```text
deteccao-fraude-cartao/
├── .gitignore
├── README.md
├── requirements.txt
├── notebooks/
│   ├── deteccao_fraudes.ipynb
│   └── observacoes_blocos.md
└── images/
    └── 13 figuras exportadas das saídas do notebook
```

| Caminho | Conteúdo |
| :--- | :--- |
| `.gitignore` | Ignora `venv/`, caches do Python/Jupyter e arquivos `.csv` baixados. |
| `README.md` | Este documento: problema, preparação, resultados, limiar, SHAP, estrutura e auditoria. |
| `requirements.txt` | Versões fixadas do ambiente local (Python 3.13). |
| `notebooks/deteccao_fraudes.ipynb` | Notebook com 66 células (48 de código e 18 de Markdown) e saídas salvas. |
| `notebooks/observacoes_blocos.md` | Guia célula a célula do notebook, com índices a partir de 0. |
| `images/` | Figuras exportadas do notebook; relação completa na Seção 11. |

O CSV não é versionado: ele é lido diretamente pela URL e `*.csv` está no `.gitignore`. O diretório `venv/` existe localmente, mas também não é versionado.

## 4. Como executar

**Local (Python 3.13):**

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1   # PowerShell (Linux/macOS: source venv/bin/activate)
python -m pip install -r requirements.txt
jupyter notebook notebooks/deteccao_fraudes.ipynb
```

**Google Colab:** abra `notebooks/deteccao_fraudes.ipynb` e instale apenas o que faltar (`xgboost` e `shap`, normalmente).

Em ambos os casos, execute as células em ordem (`Runtime > Run all`) e mantenha acesso à internet para baixar o CSV pela URL.

## 5. Preparação dos dados e Pipeline

- `df_modelo = df.copy()` preserva a base original.
- `Amount_log = np.log1p(Amount)` reduz a assimetria; a `Amount` original sai das features.
- `X` = `Time`, `V1`–`V28`, `Amount_log`; `y` = `Class` (fora de `X` para evitar *data leakage*).
- Primeira divisão estratificada: treino (70%) / teste (30%). Segunda divisão estratificada dentro do treino: treino interno (139.554 linhas, 241 fraudes) / validação (59.810 linhas, 103 fraudes). O teste (85.443 linhas, 148 fraudes) fica reservado até o fim.
- **Não há scaler manual.** Cada modelo é um `Pipeline([("scaler", StandardScaler()), ("modelo", modelo)])`, ajustado somente no conjunto de treino correspondente. Isso evita vazamento de estatísticas e permite buscar hiperparâmetros com o preprocessing embutido.
- Pesos para a fraude: `class_weight="balanced"` na regressão logística e no Random Forest; `scale_pos_weight ≈ 578` no XGBoost.

## 6. Modelos e resultados

### Seleção na validação (célula 29)

| Modelo | Precisão | Recall | F1 | ROC-AUC | AP |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Regressão Logística | 0,0683 | 0,9320 | 0,1272 | 0,9881 | 0,7759 |
| XGBoost | 0,7672 | 0,8641 | **0,8128** | 0,9857 | **0,8791** |
| Random Forest | **0,9149** | 0,8350 | 0,8731 | 0,9640 | 0,8786 |

Estes números são de **seleção**, não resultado final.

### GridSearchCV (células 38–42)

Busca em `n_estimators` [200, 300] × `max_depth` [3, 4], com `StratifiedKFold(3)`, `scoring={"recall", "precision", "f1"}` e `refit="recall"`. Melhor combinação: **200 árvores, profundidade 3**, recall médio 0,8628. O preço aparece nas outras métricas: precisão média 0,4599 e F1 0,5991. A opção 300 árvores/profundidade 3 teve recall 0,8545, mas precisão 0,6929 e F1 0,7624.

**Ressalva metodológica:** o `scale_pos_weight ≈ 578` é calculado uma única vez, antes da validação cruzada, a partir do `y_treino` completo. Consequentemente, cada fold usa um peso derivado também da proporção de classes dos próprios dados daquele fold, criando uma dependência entre o peso e a validação. Isso não invalida o estudo, mas é um ponto de aprimoramento: calcular o peso dentro de cada fold tornaria a validação cruzada mais rigorosa.

Com o limiar padrão 0,5 na validação externa, o XGBoost inicial tem precisão 0,7672 / recall 0,8641 / F1 0,8128, enquanto o otimizado tem 0,3217 / 0,8932 / 0,4730. Como as distribuições de probabilidade são diferentes, o limiar **não** foi reaproveitado do modelo inicial: ele foi selecionado depois desta etapa, com as probabilidades do `melhor_xgb`.

### Balanceamento (células 48–52)

Regressão logística usada para isolar a estratégia, sempre avaliada na validação:

| Estratégia | Precisão | Recall | F1 |
| :--- | ---: | ---: | ---: |
| `class_weight="balanced"` | 0,0683 | 0,9320 | 0,1272 |
| Undersampling | 0,0738 | 0,9320 | **0,1368** |
| Oversampling | 0,0681 | 0,9320 | 0,1269 |

O recall não mudou; undersampling e oversampling não trouxeram vantagem real e o `class_weight` segue como a opção mais simples, sem perder dados nem repetir fraudes.

### Comparação oficial no teste (célula 55)

Os três modelos foram reajustados em todo o treino e avaliados **uma única vez** no teste, com o limiar selecionado na validação para o modelo otimizado. Esta é a única tabela de comparação final — as anteriores eram de seleção, o que elimina a narrativa conflitante de duas rodadas.

| Modelo | Acurácia | Precisão | Recall | F1 | ROC-AUC | AP |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0,9994 | **0,9098** | 0,7500 | **0,8222** | 0,9429 | **0,8101** |
| XGBoost (GridSearchCV) | 0,9955 | 0,2577 | 0,8514 | 0,3956 | **0,9728** | 0,7514 |
| Regressão Logística | 0,9780 | 0,0648 | **0,8716** | 0,1206 | 0,9677 | 0,7022 |

Matriz de confusão do XGBoost final: `[[84932, 363], [22, 126]]` — 126 de 148 fraudes detectadas, com 363 transações normais investigadas. O Random Forest, por outro lado, deixa passar 37 fraudes, mas gera apenas 11 alertas falsos.

**XGBoost vs Random Forest:** o XGBoost alcançou recall maior no teste (0,8514 contra 0,7500), ou seja, encontrou mais fraudes, porém ao custo de 363 falsos positivos contra 11 do Random Forest. Isso explica por que o Random Forest teve precisão 0,9098 e F1 0,8222, muito acima dos 0,2577 e 0,3956 do XGBoost. O melhor modelo não é necessariamente o que encontra mais fraudes, mas aquele que oferece o equilíbrio mais adequado ao custo operacional: se investigar alertas é caro, o Random Forest é preferível; se o falso negativo é muito mais caro que a revisão de um alerta, o XGBoost volta a ser competitivo.

## 7. Seleção do limiar de decisão

A varredura foi movida do teste para a validação e executada **após o GridSearchCV**, usando `melhor_xgb.predict_proba(X_validacao)` — o modelo que dá origem ao classificador final. Antes da correção, o limiar vinha do XGBoost inicial, cujas probabilidades são diferentes.

| Limiar | Precisão | Recall | F1 | Falsos positivos | Falsos negativos |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0,5 | 0,3217 | 0,8932 | **0,4730** | 194 | 11 |
| 0,4 | 0,2435 | 0,9029 | 0,3835 | 289 | 10 |
| 0,3 | 0,1688 | 0,9126 | 0,2848 | 463 | 9 |
| 0,2 | 0,1083 | 0,9126 | 0,1936 | 774 | 9 |
| 0,1 | 0,0551 | 0,9320 | 0,1040 | 1.647 | 7 |

**Limiar adotado: 0,5**, escolhido pelo maior F1 **entre os limiares avaliados** (0,5; 0,4; 0,3; 0,2 e 0,1). Uma busca mais detalhada poderia encontrar outro valor com desempenho superior; a varredura aqui foi deliberadamente restrita a esses cinco pontos. Baixar para 0,3 captura apenas 2 fraudes a mais entre 103, mas eleva os falsos positivos de 194 para 463 e derruba o F1 de 0,4730 para 0,2848. Em 0,1 o recall chega a 0,9320 com 1.647 falsos alertas. Como a precisão do modelo otimizado é baixa em todos os limiares, ele só é operacionalmente adequado se a investigação de alertas for barata; nesse sentido, o Random Forest é a opção mais equilibrada.

## 8. Explicação com SHAP

`TreeExplainer` aplicado ao XGBoost final, usando os dados já transformados pelo scaler do Pipeline. Importância média absoluta (top 10):

| Variável | Importância SHAP |
| :--- | ---: |
| V4 | 1,6074 |
| V14 | 1,6052 |
| V12 | 0,5633 |
| V10 | 0,4277 |
| V11 | 0,2921 |
| V3 | 0,2233 |
| V26 | 0,2187 |
| V21 | 0,2179 |
| V7 | 0,2025 |
| V8 | 0,1739 |

O beeswarm mostra a direção e a magnitude das contribuições por transação. Como V1–V28 são componentes PCA, SHAP explica o **modelo**, não estabelece causalidade sobre a fraude.

## 9. O que mudou em relação à Expert

**Seguido:** leitura pela URL, exploração com `value_counts(normalize=True)`, `log1p`, padronização, splits estratificados, três algoritmos com peso na fraude, `classification_report`, curvas ROC e Precision-Recall, estudo de limiares e SHAP.

**Evoluções implementadas:**

1. `Pipeline(StandardScaler(), modelo)` substituiu o escalonamento manual em todas as etapas, inclusive na busca de hiperparâmetros e na explicação SHAP.
2. `GridSearchCV` com validação cruzada estratificada, escolhendo por recall e registrando precisão/F1 para expor o custo da métrica única.
3. Comparação explícita entre `class_weight`, undersampling e oversampling, pelo recall da classe fraude.
4. Rodadas de avaliação unificadas: validação para seleção, teste uma única vez para a tabela oficial.
5. Varredura de limiares movida do teste para a validação, com o limiar **selecionado após o GridSearchCV** a partir das probabilidades do XGBoost otimizado. Trata-se de seleção de limiar de decisão, não de calibração das probabilidades.

**Divergência consciente:** o roteiro sugeria adotar 0,3 para capturar mais fraudes. O limite foi testado com o modelo otimizado; como o ganho de recall foi de apenas 2 fraudes e os falsos positivos saltaram de 194 para 463, o limiar 0,5 foi mantido com justificativa na célula 46. Ainda não foram feitas variáveis de comportamento temporal nem novos algoritmos além dos três pedidos.

## 10. Auditoria dos requisitos

| Requisito | Situação | Evidência |
| :--- | :--- | :--- |
| Dataset lido pelo link da base | Atendido | Célula 2 |
| Proporção com `value_counts(normalize=True)` | Atendido | Célula 10 |
| `np.log1p`, padronização, `stratify` | Atendido | Células 17, 21 e 23 |
| Regressão logística, Random Forest e XGBoost com peso na fraude | Atendido | Células 25–27 |
| Relatório de classificação, ROC e Precision-Recall | Atendido | Células 30, 33 e 35 |
| Testar limiar 0,5 → 0,3 | Atendido | Células 44–46, com decisão justificada por 0,5 |
| `Pipeline` (padronização + modelo) | Atendido | Células 22–27 e 54 |
| `GridSearchCV` priorizando recall | Atendido | Células 38–42 |
| Undersampling / oversampling | Atendido | Células 48–52 |
| Unificar as rodadas de avaliação | Atendido | Células 53–58 |
| Limiar selecionado na validação, após o GridSearchCV, com o modelo otimizado | Atendido | Células 43–47 |
| Importância das variáveis + SHAP | Atendido | Células 59–65 |
| README com problema, preparação, modelos, limiar, SHAP, estrutura e mudanças | Atendido | Este arquivo |
| Saídas salvas (tabelas e curvas) | Atendido | 66 células (48 de código / 18 de Markdown); 43 com contador de execução, 44 com saídas salvas, 13 figuras e nenhuma saída `error` |
| Dataset fora do repositório | Atendido | Nenhum CSV versionado; `*.csv` está no `.gitignore` |
| Ambiente e dependências documentados | Atendido | Seção 4 e `requirements.txt` |
| Organização de pastas revisada | Atendido | Seção 3; `venv/`, caches e CSVs ignorados pelo `.gitignore` |
| Todo arquivo citado no README existe | Atendido | `README.md`, `requirements.txt`, `.gitignore`, os dois arquivos de `notebooks/` e as 13 figuras de `images/` |
| Notebook sem erros registrados | Atendido | Nenhuma saída `error` no arquivo salvo; reexecução exige internet para baixar o CSV |

## 11. Organização, ambiente e figuras

- `venv/`, `__pycache__/`, `.ipynb_checkpoints/` e `*.csv` ficam fora do controle de versão; apenas o código, as anotações, as dependências e as figuras são versionados.
- `notebooks/observacoes_blocos.md` usa índices de célula a partir de 0, exatamente como o arquivo `.ipynb` salvo.
- As figuras em `images/` são exportações das saídas do notebook. Se um gráfico mudar, ele deve ser reexportado para manter esta galeria sincronizada.
- O arquivo de dispersão SHAP com as 10 variáveis mais importantes foi renomeado de `isualizando...` para `Visualizando a importância das features com SHAP (top 10).png`.

### Exploração e preparação

![Distribuição de transações normais x fraudulentas](<images/Distribuição de transações normais x fraudulentas.png>)

![Distribuição dos valores das transações](<images/Distribuição dos valores das transações.png>)

![Distribuição original dos valores](<images/Distribuição original dos valores.png>)

![Comparando a distribuição antes e depois da transformação logarítmica](<images/Comparando a distribuição antes e depois da transformação logarítmica.png>)

### Seleção e avaliação dos modelos

![Visualizando as matrizes de confusão dos modelos na validação](<images/Visualizando as matrizes de confusão dos modelos na validação.png>)

![Curva ROC - validação](<images/Curva ROC - validação.png>)

![Curva Precision-Recall - validação](<images/Curva Precision-Recall - validação.png>)

![Comparação das métricas por limiar (modelo otimizado)](<images/Comparação das métricas por limiar (modelo otimizado).png>)

![Estratégias de balanceamento - validação](<images/Estratégias de balanceamento - validação.png>)

![Comparação final dos modelos no teste](<images/Comparação final dos modelos no teste.png>)

### Explicabilidade

![Visualizando a importância das features com SHAP](<images/Visualizando a importância das features com SHAP.png>)

![Visualizando a importância das features com SHAP (gráfico de dispersão)](<images/Visualizando a importância das features com SHAP (gráfico de dispersão).png>)

![Visualizando a importância das features com SHAP (top 10)](<images/Visualizando a importância das features com SHAP (top 10).png>)
