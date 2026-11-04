@echo off
REM ------------------------------------------------------------------
REM  Ejecuta todo el pipeline NPE en orden y se detiene si hay un error.
REM  Copie este archivo dentro de la carpeta "scripts" y haga doble clic,
REM  o ejecútelo desde la ventana de comandos (cmd) estando en esa carpeta.
REM ------------------------------------------------------------------
chcp 65001 > nul
cd /d "%~dp0"

echo Inicio: %date% %time%
for %%S in (00_check_env 01_audit_data 02_build_dataset 03_descriptives 04_multilevel 05_models 06_unseen_universities 07_contextualized_npe 08_fairness_audit 09_explainability 10_sensitivity 12_temporal_validation 13_university_stability 11_export_latex) do (
    echo.
    echo ==================== %%S ====================
    python %%S.py
    if errorlevel 1 (
        echo.
        echo ERROR en %%S.py. Revise el log en outputs\logs\%%S.log
        pause
        exit /b 1
    )
)
echo.
echo Fin: %date% %time%
echo TODO TERMINO CORRECTAMENTE.
pause
