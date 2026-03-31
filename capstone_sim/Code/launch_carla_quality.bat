@echo off
REM Launch CARLA with high quality rendering settings
REM This fixes "blob" cars and improves visual quality

cd /d C:\Users\omarj\Desktop\Capstone_Work\capstone_sim\CARLA_0.9.16

echo Starting CARLA with MAXIMUM quality settings...
echo This will disable LOD and show full vehicle detail at all distances
echo.

start CarlaUE4.exe -quality-level=Epic -ResX=1920 -ResY=1080 -windowed -benchmark -fps=30

REM The -benchmark flag disables some optimizations that reduce vehicle quality
REM -fps=30 locks framerate for consistency

REM Alternative: For even better quality but slower performance
REM CarlaUE4.exe -quality-level=Epic -ResX=1920 -ResY=1080 -windowed -dx12

REM For maximum quality (very slow):
REM CarlaUE4.exe -quality-level=Cinematic -ResX=2560 -ResY=1440 -windowed
