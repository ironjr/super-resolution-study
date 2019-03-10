@ECHO OFF

SET TODAY=%DATE:~2,2%%DATE:~5,2%%DATE:~8,2%
SET DIRNAME=archive/%TODAY%

SET /A "ITER=0"
:LOOP
IF EXIST %DIRNAME%-%ITER%\* (
    SET /A "ITER=%ITER% + 1"
    GOTO :LOOP
)
SET DIRNAME=%DIRNAME%-%ITER%

IF NOT EXIST .\%DIRNAME%\* MKDIR .\%DIRNAME%
IF NOT EXIST .\%DIRNAME%\logs\* MKDIR .\%DIRNAME%\logs
MOVE .\checkpoints\* .\%DIRNAME%
MOVE .\logs\* .\%DIRNAME%\logs
xcopy /S .\schedule.json ".\%DIRNAME%"
IF NOT EXIST .\checkpoints MKDIR .\checkpoints
IF NOT EXIST .\logs MKDIR .\logs