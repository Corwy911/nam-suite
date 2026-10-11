@echo off
rem Build della NAM Suite: lancia suite\build.py passando tutti gli argomenti.
rem Gli installer compilati finiscono nella cartella 03-installer accanto al repo,
rem grazie a NAM_SUITE_OUT (letta da build.py al posto di suite\output).
rem
rem Uso:
rem   build-installer.cmd all              build completa (tools, launcher, runtime, trainer, varianti, capture, extras, iss)
rem   build-installer.cmd trainer          ri-stagia solo il trainer (dopo modifiche a trainer\nam-web)
rem   build-installer.cmd iss              ricompila solo l'installatore con Inno Setup
rem   build-installer.cmd iss --variants cpu   build di prova: solo la variante CPU
rem
rem La cache di build resta in suite\cache: non cancellarla, rende le build veloci (zero download).

setlocal
set "NAM_SUITE_OUT=%~dp0..\03-installer"
python "%~dp0suite\build.py" %*
endlocal
