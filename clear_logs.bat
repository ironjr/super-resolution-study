@ECHO OFF

CD /D .\checkpoints
FOR /F "DELIMS=" %%I IN ('DIR /B') DO (RMDIR "%%I" /S /Q || DEL "%%I" /S /Q)
CD /D ..\logs
FOR /F "DELIMS=" %%I IN ('DIR /B') DO (RMDIR "%%I" /S /Q || DEL "%%I" /S /Q)