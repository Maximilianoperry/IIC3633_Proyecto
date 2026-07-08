# Sistema de Recomendación Conversacional (CRS) Basado en RAG

Este repositorio contiene la implementación de un **Sistema de Recomendación Conversacional (CRS)** para películas, basado en **Generación Aumentada por Recuperación (RAG)**. El sistema representa el historial de diálogo mediante *embeddings* densos (Sentence-BERT), recupera conversaciones de entrenamiento similares y genera recomendaciones en lenguaje natural utilizando un Modelo de Lenguaje de Gran Escala (LLM).

El proyecto se evalúa sobre el dataset **ReDial** (Recommender Dialogue Dataset), comparando el desempeño de la propuesta frente a *baselines* clásicos (Random, Most Popular, BM25) y modelos basados en LLM reportados en la literatura.

> **Nota:** GitHub puede mostrar el error *"Invalid Notebook"* al intentar previsualizar el archivo `.ipynb` directamente en el navegador. Esto se debe a metadata de widgets de progreso (generada por Colab al descargar los modelos) que no cumple con el validador de vista previa de GitHub, y no afecta el contenido ni la funcionalidad del notebook. **Para ver el código, basta con descargar el archivo** y abrirlo en Jupyter, Google Colab o VS Code.

## Contenido del Notebook

El notebook `Sistemas_de_Recomendacion_Conversacional.ipynb` está organizado en las siguientes secciones:

1. **Configuración y Setup**
   Montaje de Google Drive, descompresión del dataset ReDial y carga de *embeddings* cacheados (para evitar recomputarlos en cada ejecución).

2. **Instalación de Librerías Esenciales**
   Instalación de `pandas`, `numpy`, `rank_bm25` y `sentence-transformers`.

3. **Procesamiento y Análisis de Datos**
   - Carga y parseo de `train_data.jsonl` / `test_data.jsonl`.
   - Cálculo de estadísticas descriptivas del dataset (total de conversaciones, turnos, películas mencionadas, etc.).
   - Análisis del fenómeno de **sesgo de popularidad (long-tail)**: concentración de menciones en un pequeño porcentaje de películas.
   - Visualización de la distribución de turnos por conversación y de la distribución *long-tail* de películas mencionadas.

4. **Evaluación de Modelos Baseline**
   Implementación y evaluación de tres modelos de referencia:
   - **Random**: recomendación aleatoria.
   - **Most Popular**: recomienda los ítems más mencionados en el dataset.
   - **Retrieval BM25**: recuperación léxica clásica sobre la metadata de las películas.

   Métricas utilizadas: **HR@10** (Hit Rate) y **MRR@10** (Mean Reciprocal Rank).

5. **Solución Propuesta (RAG)**
   - Carga del modelo **Sentence-BERT** (`all-MiniLM-L6-v2`) para generar *embeddings* densos.
   - Generación y almacenamiento de *embeddings* de las conversaciones de entrenamiento y de prueba, variando el tamaño de la ventana de contexto ($N$ turnos considerados).
   - Recuperación de las $K$ conversaciones de entrenamiento más similares mediante similitud del coseno.
   - Agregación de películas candidatas según su frecuencia de aparición en las conversaciones recuperadas.
   - **Análisis de sensibilidad**: barrido sobre distintos valores de $N$ (tamaño de contexto: 1, 2, 4, 8, 16, completo) y $K$ (número de conversaciones similares: 5, 10, 20, 50, 100, 200), generando gráficos comparativos de HR@10 y MRR@10.

6. **Generación de Respuesta en Lenguaje Natural**
   - Carga de un modelo generativo (`Qwen/Qwen3-4B-Instruct-2507`) mediante `transformers`.
   - Implementación de la clase `RAGRecommender`, que integra el modelo de recuperación (Sentence-BERT) y el LLM generativo para producir sugerencias conversacionales basadas en las películas recuperadas.
   - Pruebas del sistema completo sobre diálogos de ejemplo del conjunto de *test*.

## Requisitos

```bash
pip install pandas numpy
pip install rank_bm25
pip install sentence-transformers
pip install transformers torch
```

El notebook está diseñado para ejecutarse en **Google Colab**, con el dataset almacenado en Google Drive (`/content/drive/MyDrive/RecSys/redial_dataset.zip`). Si se ejecuta en un entorno local, deben ajustarse las rutas de montaje de Drive y de carga del dataset.

## Dataset

Se utiliza el dataset **ReDial** (Li et al., 2018), compuesto por conversaciones de recomendación de películas en lenguaje natural, con anotaciones explícitas de películas mencionadas y respuestas de aceptación/rechazo por parte del usuario.

- `train_data.jsonl`: conjunto de entrenamiento, usado para construir el catálogo de películas, calcular popularidad y generar los *embeddings* de referencia.
- `test_data.jsonl`: conjunto de prueba, usado para evaluar el desempeño de los modelos.

## Métricas de Evaluación

- **HR@10 (Hit Rate)**: proporción de casos en que al menos una película relevante aparece dentro del top-10 de recomendaciones.
- **MRR@10 (Mean Reciprocal Rank)**: promedio del inverso de la posición de la primera película relevante dentro del top-10.

## Resultados Principales

| Modelo | HR@10 | MRR@10 |
|---|---|---|
| Random | 0.00142 | 0.0036 |
| Most Popular | 0.2399 | 0.1214 |
| Retrieval BM25 | 0.3271 | 0.1755 |
| GPT-2 (Qiu et al., 2025) | 0.147 | 0.051 |
| Llama3.1-8B (Qiu et al., 2025) | 0.188 | 0.078 |
| **Ours (N=Full, K=50)** | **0.6363** | **0.3599** |

La configuración óptima encontrada mediante el análisis de sensibilidad corresponde a utilizar el contexto conversacional completo ($N=\text{Full}$) junto con $K=50$ conversaciones similares recuperadas.

> **Nota:** si bien el sistema propuesto supera ampliamente a los *baselines* y modelos LLM comparados, este resultado debe interpretarse considerando el fuerte sesgo de popularidad presente en el dataset, el cual podría explicar parte de la mejora observada más allá de una comprensión semántica superior del diálogo.

## Autores

- Maximiliano Perry
- Jenaro Prieto

Pontificia Universidad Católica de Chile — Departamento de Ciencias de la Computación
Curso: IIC3633 (Sistemas Recomendadores) — Profesor: Denis Parra

## Referencias

- Li, R., Kahou, S. E., Adi, H., Bateni, C., Pascanu, R., and Courville, A. *Towards deeply-conversational recommendations*. NeurIPS, 2018.
- Qiu, Z., Luo, L., Zhao, Z., Pan, S., and Liew, A. W.-C. *Graph retrieval-augmented LLM for conversational recommendation systems*. arXiv:2503.06430, 2025.
- Google. *Gemini* (modelo de lenguaje). Utilizado como herramienta de apoyo para la generación y depuración de código en el presente proyecto.
