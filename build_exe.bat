@echo off
REM Construit AutoPreprocessor.exe (double-clic, sans Python installe sur la machine cible)
REM A lancer depuis le dossier du projet, de preference HORS d'OneDrive (ex. C:\AutoPre)
REM (la ou se trouvent app_preprocess.py et auto_preprocessor.py)

if not exist app_preprocess.py (
  echo ERREUR : app_preprocess.py introuvable dans ce dossier.
  pause
  exit /b 1
)
if not exist auto_preprocessor.py (
  echo ERREUR : auto_preprocessor.py introuvable dans ce dossier.
  pause
  exit /b 1
)

py -m pip install --upgrade pyinstaller openpyxl
if errorlevel 1 (
  echo ERREUR pendant l'installation de PyInstaller.
  pause
  exit /b 1
)

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

py -m PyInstaller --noconfirm --clean --onefile --windowed --name AutoPreprocessor ^
  --hidden-import sklearn.utils._typedefs --hidden-import sklearn.neighbors._partition_nodes ^
  --collect-submodules sklearn --collect-submodules scipy ^
  app_preprocess.py
if errorlevel 1 (
  echo.
  echo ERREUR : la construction a echoue. Copiez les dernieres lignes ci-dessus.
  pause
  exit /b 1
)

echo.
if not exist dist\AutoPreprocessor.exe (
  echo ERREUR : dist\AutoPreprocessor.exe n'existe pas.
  pause
  exit /b 1
)
powershell -NoProfile -Command "$s=(Get-Item 'dist\AutoPreprocessor.exe').Length/1MB; if($s -lt 20){Write-Host ('ATTENTION : fichier trop petit ({0:N2} Mo), build probablement incomplet.' -f $s)} else {Write-Host ('OK : dist\AutoPreprocessor.exe ({0:N0} Mo)' -f $s)}"
pause