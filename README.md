# npe-benchmark

Código del artículo de Barturén, Musayon y Tocto sobre la heterogeneidad institucional de la nota promedio al egreso (NPE) y la generalización de su estimación retrospectiva, con registros TUNI–SUNEDU de los cortes 2022–2025 (envío a *Education Sciences*). Todo el código está en Python.

## 1. Instalación (una sola vez)

Se usó **Python 3.10.11** en Windows 10. Las versiones exactas de todas las librerías están en `requirements_congelado.txt`.

```bash
cd npe
python -m venv venv
venv\Scripts\activate              # Linux/Mac: source venv/bin/activate
pip install -r requirements_congelado.txt
```

## 2. Preparar los datos

1. Copiar los cuatro cortes de TUNI a `data/raw/` como `tuni_2022.csv`, `tuni_2023.csv`, `tuni_2024.csv` y `tuni_2025.csv` (o cambiar las rutas en `config.yaml`). Los archivos originales de TUNI usan separador `|` y codificación `latin-1`.
2. Revisar en `config.yaml`:
   - `read`: separador, codificación y decimal.
   - `columns`: el nombre exacto de cada columna en los archivos.
   - `filters`: tipo de entidad (`UNIVERSIDAD`) y nivel (`CARRERA PROFESIONAL`).
   - `anio_referencia_edad`: año en que se descargaron los datos (2026), usado para calcular la edad al egreso.
3. Opcional: crear `data/external/nota_minima_universidad.csv` con las columnas `universidad,nota_minima,fuente`.

La descripción de cada variable (columna de TUNI, tipo, valores y uso en el estudio) está en `diccionario_datos.csv`.

Para comprobar que los archivos son los mismos que se usaron en el artículo, comparar sus huellas con `huellas_entrada.txt`:

```bash
certutil -hashfile data\raw\tuni_2022.csv SHA256
```

## 3. Ejecutar

```bash
cd src
correr_todo.bat                    # ejecuta 00 a 13 en orden y se detiene si hay un error
```

Si la ejecución se interrumpe después del script 04, se puede continuar con `continuar_desde_05.bat`.

Cada script deja su log en `outputs/logs/`, sus tablas en `outputs/tables/` y sus figuras en `outputs/figures/`.

| Script | Qué hace | Salida en el artículo | Tiempo aproximado* |
|---|---|---|---|
| 00_check_env | Verifica librerías y columnas | — | < 1 min |
| 01_audit_data | Flujo de depuración por corte | Figura 2 | 1 min |
| 02_build_dataset | Base analítica, partición por persona y huellas SHA-256 | — | 1 min |
| 03_descriptives | Descriptivos | Tabla 2, Figura 3 | < 1 min |
| 04_multilevel | Componentes de varianza y CCI (M0–M2) | Tablas 3 y 4, Figura 4 | 20–35 min |
| 05_models | Referencias, modelos mixtos y aprendizaje automático | Tabla 5, Figura 5 | 6–15 h |
| 06_unseen_universities | Universidades no observadas | Tabla 6, Figura S1 | 1,5–2 h |
| 07_contextualized_npe | NPE contextualizada | Tabla 10, Figura 7 | 5 min |
| 08_fairness_audit | Error y sesgo por subgrupos | Tabla 9, Figura S2 | < 1 min |
| 09_explainability | SHAP y permutación por bloques | Tabla 8, Figura 6 | < 1 min |
| 10_sensitivity | Sensibilidad: créditos inválidos y redondeo | Tabla S3 | 35 min |
| 12_temporal_validation | Entrenar con 2022–2024 y evaluar en 2025 | Tabla 7, Figura S3 | 20 min |
| 13_university_stability | Estabilidad del efecto universidad y tendencia | Figura S4 | 5 min |
| 11_export_latex | Exporta 27 tablas, 9 figuras y 71 cifras clave | — | < 1 min |

\*Tiempos reales de las corridas canónicas con 433 213 registros, en una laptop con `n_jobs: 4`. El afinamiento de Random Forest y CatBoost (script 05) es el paso más lento.

## 4. Reproducibilidad

- El procedimiento completo se ejecutó **dos veces desde cero** (corridas del 1–2 y 2–3 de noviembre de 2026) y produjo resultados idénticos.
- El script 02 genera la partición y la guarda en `data/processed/manifiesto_particion.csv` (no se publica porque contiene un registro por egresado). Para verificarla, basta comparar sus cuatro huellas SHA-256 (entrenamiento, validación, prueba y pliegues por universidad) con las de `resultados_corrida_1/logs/02_build_dataset.log`.
- Las huellas de los datos de entrada y del código están en `huellas_entrada.txt` y `huellas_codigo.txt`.
- Semilla fija: `20261021` (en `config.yaml`).
- Las tablas, figuras y logs de la corrida canónica (corrida 1) están en `resultados_corrida_1/`.

Equivalencia entre los archivos de `resultados_corrida_1/figures/` y el artículo:

| Archivo | Artículo |
|---|---|
| fig02_distribucion_npe | Figura 3 |
| fig03_componentes_varianza | Figura 4 |
| fig04_comparacion_modelos | Figura 5 |
| fig05_universidades_no_vistas | Figura S1 |
| fig06_npe_contextualizada | Figura 7 |
| fig07_auditoria_subgrupos | Figura S2 |
| fig08_explicabilidad | Figura 6 |
| fig09_validacion_temporal | Figura S3 |
| fig10_estabilidad_tendencia | Figura S4 |

La Figura 2 (flujo de depuración) se elaboró a partir de `tables/01_flujo_por_corte.csv` y `tables/02_flujo_final.csv`.

## 5. Reglas de diseño (para Métodos)

- El preprocesamiento y la imputación se ajustan solo con el conjunto de entrenamiento. La prueba se evalúa una sola vez, después de seleccionar el modelo con validación.
- Las particiones se agrupan por persona. Los IC son bootstrap agrupados por universidad.
- La NPE contextualizada usa **solo universidad, programa y año**. Sexo, edad y origen nunca ajustan notas; se usan para auditar.

## 6. Datos personales

`data/raw/` contiene el identificador de persona y no se sube a GitHub (ver `.gitignore`). La base analítica reemplaza ese identificador por un código interno no reversible.

## 7. Uso de IA

- **Herramienta:** Claude (Anthropic, modelo Claude Opus 5.5)
- **Fecha de uso:** 30/10/2026 – 03/11/2026

### 1. Preparación de los datos
- **Para qué:** verificar la calidad de los datos y corregir el código durante este proceso
- **Verificación:** se compararon los conteos del flujo de depuración con los archivos originales de TUNI y se ejecutó el pipeline dos veces desde cero
- **Decisión:** se adoptó la versión corregida del script de construcción de la base

### 2. Modelado
- **Para qué:** revisar y corregir el código de los modelos durante este proceso

### 3. Comunicación
- **Para qué:** revisar la concordancia de los párrafos del manuscrito
