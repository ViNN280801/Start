@echo on

pyinstaller --noconfirm ^
            --onedir ^
            --console ^
            --clean ^
            --log-level "INFO" ^
            --name "nia_start_exe" ^
            --add-data "build;build/" ^
            --add-data "icons;icons/" ^
            --add-data "results;results/" ^
            --add-data "tests;tests/" ^
            --add-data "ui;ui/" ^
            --add-binary "C:/Users/vladislavsemykin/.conda/envs/startenv/Library/bin/*.dll;." ^
            --add-binary "C:/Users/vladislavsemykin/.conda/envs/startenv/Lib/site-packages/vtk.libs/*.dll;." ^
            --add-binary "Release/nia_start_core.exe;." ^
            --add-binary "Release/gmsh-4.12.dll;." ^
            --paths "ui" ^
            "ui/main.py"

pause
