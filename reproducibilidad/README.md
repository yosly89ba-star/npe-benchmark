# Reproducibilidad

El análisis completo se ejecutó dos veces de forma independiente con la misma semilla y el mismo entorno (Python 3.10.11, Windows).
El archivo de cifras numbers.json de ambas corridas tiene la misma huella SHA-256:

| Corrida | Archivo | SHA-256 |
|---|---|---|
| 1 | resultados_corrida_1/numbers.json | DDB5C89F2C77C93D1E07EFE88A2D0024C1272CF2C473F3D096B54B35FC8326A4 |
| 2 | reproducibilidad/corrida_2/numbers.json | DDB5C89F2C77C93D1E07EFE88A2D0024C1272CF2C473F3D096B54B35FC8326A4 |

Para comprobarlo en Windows: Get-FileHash <archivo>
