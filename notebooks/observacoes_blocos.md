# Observações por bloco — `deteccao_fraudes.ipynb`

Este arquivo descreve o que cada bloco (célula) do notebook [`deteccao_fraudes.ipynb`](deteccao_fraudes.ipynb) faz. Os números seguem a ordem do notebook, começando em 0, e incluem células de código e de Markdown. A ordem foi ajustada para que o limiar seja **selecionado** depois do `GridSearchCV`, com as probabilidades do modelo otimizado. Trata-se de seleção de limiar de decisão, não de calibração das probabilidades.

## 1. Importações e exploração (células 0–13)

- **Célula 0 (código)** — Importa `pandas`, `numpy`, `matplotlib` e `seaborn` para tabelas, cálculos e visualizações.
- **Célula 1 (Markdown)** — Abre a seção "Explorando os Dados".
- **Célula 2 (código)** — Carrega o CSV diretamente da URL oficial com `pd.read_csv(url)` e mostra `df.head()`. O arquivo não é salvo localmente.
- **Célula 3 (código)** — Mostra o formato da base: `(284807, 31)`.
- **Célula 4 (Markdown)** — Introduz a checagem de valores ausentes.
- **Célula 5 (código)** — `df.info()` mostra tipos e contagens não nulas de `Time`, `V1`–`V28`, `Amount` e `Class`.
- **Célula 6 (código)** — `df.describe()` traz estatísticas descritivas das variáveis numéricas.
- **Célula 7 (código)** — `df.isnull().sum()` conta nulos por coluna.
- **Célula 8 (código)** — `df.isnull().sum().sum()` confirma que não há nenhum valor nulo (0).
- **Célula 9 (código)** — `value_counts()` mostra 284.315 normais e 492 fraudes.
- **Célula 10 (código)** — `value_counts(normalize=True) * 100` quantifica o desbalanceamento: 99,8273% normais e 0,1727% fraudes.
- **Célula 11 (código)** — `countplot` de `Class`: a barra de fraudes quase não aparece.
- **Célula 12 (código)** — Histograma de `Amount`, com forte assimetria à direita.
- **Célula 13 (código)** — `groupby("Class")["Amount"].describe()` compara valores entre normais e fraudulentas.

## 2. Preparação e divisão dos dados (células 14–22)

- **Célula 14 (Markdown)** — Abre a preparação dos dados.
- **Célula 15 (código)** — Cria `df_modelo = df.copy()` para preservar a base original.
- **Célula 16 (código)** — Plota o histograma original de `Amount` como referência.
- **Célula 17 (código)** — Cria `Amount_log = np.log1p(Amount)` para suavizar a distribuição.
- **Célula 18 (código)** — Compara lado a lado `Amount` e `Amount_log`.
- **Célula 19 (código)** — Separa `X` (remove `Class` e `Amount`, mantém `Amount_log`) e `y = Class`.
- **Célula 20 (Markdown)** — Explica `X`, `y`, o vazamento evitado ao retirar `Class` e a substituição de `Amount`.
- **Célula 21 (código)** — Faz duas divisões estratificadas: treino/teste (70/30) e, dentro do treino, treino interno/validação (70/30). Saídas: treino interno com 139.554 linhas e 241 fraudes; validação com 59.810 linhas e 103 fraudes; teste reservado com 85.443 linhas e 148 fraudes.
- **Célula 22 (Markdown)** — Explica que não existe scaler manual: o `StandardScaler` vive dentro do `Pipeline`.

## 3. Pipeline e modelos candidatos (células 23–33)

- **Célula 23 (código)** — Importa `Pipeline`, `StandardScaler`, modelos, métricas, `StratifiedKFold`, `GridSearchCV`, `clone` e XGBoost; define `construir_pipeline(modelo)` com as etapas `scaler` e `modelo`.
- **Célula 24 (código)** — Calcula `peso_fraude = 139313 / 241 ≈ 578,06` no treino interno.
- **Célula 25 (código)** — Ajusta a regressão logística com `class_weight="balanced"` dentro do pipeline.
- **Célula 26 (código)** — Ajusta o Random Forest (100 árvores, `class_weight="balanced"`).
- **Célula 27 (código)** — Ajusta o XGBoost inicial (300 árvores, profundidade 4, `scale_pos_weight`).
- **Célula 28 (código)** — Define `avaliar_modelo()`, que aplica um limiar às probabilidades e retorna acurácia, precisão, recall, F1, ROC-AUC e AP.
- **Célula 29 (código)** — Compara os três candidatos **na validação**: LR (precisão 0,0683; recall 0,9320; F1 0,1272), XGBoost inicial (0,7672; 0,8641; 0,8128) e RF (0,9149; 0,8350; 0,8731). É uma tabela de seleção.
- **Célula 30 (código)** — Imprime `classification_report` dos candidatos na validação.
- **Célula 31 (código)** — Plota as três matrizes de confusão da validação.

## 4. Curvas na validação (células 32–37)

- **Célula 32 (Markdown)** — Explica que ROC e Precision-Recall usam probabilidades da validação.
- **Célula 33 (código)** — Plota ROC com AUCs: LR 0,9881, XGBoost 0,9857, RF 0,9640.
- **Célula 34 (Markdown)** — Explica AUC e alerta que ele não substitui precisão/recall.
- **Célula 35 (código)** — Plota Precision-Recall com APs: XGBoost 0,8791, RF 0,8786, LR 0,7759.
- **Célula 36 (Markdown)** — Esclarece que essas tabelas são de seleção, não a comparação final.
- **Célula 37 (código)** — Tabela ordenada de ROC-AUC e AP na validação.

## 5. GridSearchCV (células 38–42)

- **Célula 38 (Markdown)** — Explica `modelo__` no `param_grid`, validação cruzada estratificada e `scoring` múltiplo com `refit="recall"`.
- **Célula 39 (código)** — Busca `n_estimators` [200, 300] × `max_depth` [3, 4] em 3 folds. Melhor combinação: 200 árvores, profundidade 3, recall médio 0,8628.
- **Célula 40 (código)** — Tabela com recall, precisão e F1 médios: o melhor recall tem precisão 0,4599 e F1 0,5991; a combinação 300/3 tem recall 0,8545, precisão 0,6929 e F1 0,7624.
- **Célula 41 (código)** — Cria `melhor_xgb = grid.best_estimator_` e guarda `probabilidades_grid` na validação. Compara, com o limiar padrão 0,5, o XGBoost inicial (precisão 0,7672 / recall 0,8641 / F1 0,8128) com o otimizado (0,3217 / 0,8932 / 0,4730).
- **Célula 42 (Markdown)** — Interpreta o custo de otimizar apenas recall e anuncia que o limiar será selecionado com as probabilidades do modelo otimizado.

## 6. Limiar de decisão com o modelo otimizado (células 43–47)

- **Célula 43 (Markdown)** — Justificativa metodológica da correção: o GridSearchCV altera a distribuição das probabilidades, então o limiar deve ser escolhido com `melhor_xgb`, não com o modelo inicial. O teste segue reservado.
- **Célula 44 (código)** — Usa `probabilidades_grid = melhor_xgb.predict_proba(X_validacao)[:, 1]`, varre 0,5 a 0,1, registra precisão, recall, F1, FP e FN, e escolhe `LIMIAR_ESCOLHIDO = 0,5` pelo maior F1 entre os limiares avaliados (0,4730). Com 0,3 o recall sobe para 0,9126, mas os falsos positivos vão a 463 e o F1 cai para 0,2848; em 0,1 são 1.647 falsos positivos.
- **Célula 45 (código)** — Gráfico de precisão, recall e F1 por limiar para o modelo otimizado.
- **Célula 46 (Markdown)** — Interpreta o trade-off com os números do modelo otimizado e observa que sua precisão é baixa em qualquer limiar.
- **Célula 47 (código)** — Relatório de classificação e matriz de confusão na validação com o limiar escolhido: `[[59513, 194], [11, 92]]`.

## 7. Balanceamento (células 48–52)

- **Célula 48 (Markdown)** — Apresenta `class_weight`, undersampling e oversampling, comparados pelo recall.
- **Célula 49 (código)** — Undersampling reduz a maioria a 241 linhas (total 482); oversampling replica fraudes até 139.313 (total 278.626); ajusta um pipeline de regressão logística para cada caso.
- **Célula 50 (código)** — Tabela: as três estratégias empatam em recall (0,9320); F1 de 0,1368 (undersampling), 0,1272 (`class_weight`) e 0,1269 (oversampling).
- **Célula 51 (código)** — Gráfico de barras das três estratégias.
- **Célula 52 (Markdown)** — Conclui que o recall não mudou e que `class_weight` segue como a opção mais simples.

## 8. Modelos finais e avaliação única no teste (células 53–58)

- **Célula 53 (Markdown)** — Define que os modelos serão reajustados em todo o treino e avaliados uma única vez no teste, com o limiar selecionado na validação para o modelo otimizado.
- **Célula 54 (código)** — Recalcula o peso final (≈578,16), clona o melhor XGBoost do GridSearchCV e reajusta LR, RF e XGBoost em `X_train`.
- **Célula 55 (código)** — Tabela oficial no teste: RF (precisão 0,9098; recall 0,7500; F1 0,8222; AUC 0,9429; AP 0,8101), XGBoost GridSearchCV (0,2577; 0,8514; 0,3956; 0,9728; 0,7514) e LR (0,0648; 0,8716; 0,1206; 0,9677; 0,7022).
- **Célula 56 (código)** — Relatório e matriz de confusão do XGBoost final no teste: `[[84932, 363], [22, 126]]` — 126 de 148 fraudes detectadas e 363 falsos alertas.
- **Célula 57 (código)** — Gráfico de barras da comparação final.
- **Célula 58 (Markdown)** — Narrativa unificada: RF é melhor para reduzir falsos positivos; XGBoost captura mais fraudes ao custo de precisão; a escolha depende do custo de cada erro.

## 9. Explicação com SHAP (células 59–65)

- **Célula 59 (código)** — Importa `shap`, transforma `X_test` com o scaler do pipeline final (`X_test_transformado`) e calcula `shap_values` com `TreeExplainer` no modelo XGBoost.
- **Célula 60 (código)** — Gráfico de barras da importância média absoluta.
- **Célula 61 (código)** — Beeswarm com todas as variáveis.
- **Célula 62 (código)** — Top 10: V4 (1,6074), V14 (1,6052), V12 (0,5633), V10 (0,4277), V11 (0,2921), V3 (0,2233), V26 (0,2187), V21 (0,2179), V7 (0,2025) e V8 (0,1739).
- **Célula 63 (Markdown)** — Explica a importância média e a limitação das componentes PCA.
- **Célula 64 (código)** — Beeswarm limitado a 10 variáveis.
- **Célula 65 (Markdown)** — Interpreta a direção das contribuições e reforça que SHAP não estabelece causalidade.
