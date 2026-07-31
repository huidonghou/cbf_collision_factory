@echo off
set LAUNCH=python .\franka_v2.py
set COMMON=--num_samples 200 --headless

:: %LAUNCH% %COMMON% --condition nominal --t_grasp 99 --t_switch 99 --t_end 24 --seed 0
%LAUNCH% %COMMON% --condition unknown --t_grasp 12 --t_switch 99 --t_end 30 --seed 0
%LAUNCH% %COMMON% --condition unknown --t_grasp 12 --t_switch 99 --t_end 30 --seed 1
%LAUNCH% %COMMON% --condition unknown --t_grasp 12 --t_switch 99 --t_end 30 --seed 2
echo Wave 1 complete.