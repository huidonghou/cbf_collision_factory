@echo off
set LAUNCH=python .\franka_v2.py
set COMMON=--num_samples 200 --headless --t_end 30

REM ==== CORE (the item-4 matrix) ====
REM -- switch-only: nominal/L1-off (metric-artifact panel) --
%LAUNCH% %COMMON% --condition nominal --l1_off --t_grasp 99 --t_switch 17 --seed 0
%LAUNCH% %COMMON% --condition nominal --l1_off --t_grasp 99 --t_switch 17 --seed 1
%LAUNCH% %COMMON% --condition nominal --l1_off --t_grasp 99 --t_switch 17 --seed 2
REM -- switch-only: unknown/L1-on (L1-doesn't-meddle) --
%LAUNCH% %COMMON% --condition unknown --t_grasp 99 --t_switch 17 --seed 0
%LAUNCH% %COMMON% --condition unknown --t_grasp 99 --t_switch 17 --seed 1
%LAUNCH% %COMMON% --condition unknown --t_grasp 99 --t_switch 17 --seed 2
REM -- separated: unknown/L1-on (the payoff) --
%LAUNCH% %COMMON% --condition unknown --t_grasp 12 --t_switch 17 --seed 0
%LAUNCH% %COMMON% --condition unknown --t_grasp 12 --t_switch 17 --seed 1
%LAUNCH% %COMMON% --condition unknown --t_grasp 12 --t_switch 17 --seed 2
REM -- simultaneous replica: unknown/L1-on --
%LAUNCH% %COMMON% --condition unknown --t_grasp 12 --t_switch 12 --seed 0
%LAUNCH% %COMMON% --condition unknown --t_grasp 12 --t_switch 12 --seed 1
%LAUNCH% %COMMON% --condition unknown --t_grasp 12 --t_switch 12 --seed 2
REM -- rung-2 completion: nominal/L1-on holds --
%LAUNCH% %COMMON% --condition nominal --t_grasp 99 --t_switch 99 --seed 1
%LAUNCH% %COMMON% --condition nominal --t_grasp 99 --t_switch 99 --seed 2

REM ==== EXTENSION (the L1-off failure family, decoupled) ====
REM -- simultaneous: unknown/L1-off (the 0.839 cell, on plant B) --
%LAUNCH% %COMMON% --condition unknown --l1_off --t_grasp 12 --t_switch 12 --seed 0
%LAUNCH% %COMMON% --condition unknown --l1_off --t_grasp 12 --t_switch 12 --seed 1
%LAUNCH% %COMMON% --condition unknown --l1_off --t_grasp 12 --t_switch 12 --seed 2
REM -- grasp-only: unknown/L1-off (does failure need the maneuver?) --
%LAUNCH% %COMMON% --condition unknown --l1_off --t_grasp 12 --t_switch 99 --seed 0
%LAUNCH% %COMMON% --condition unknown --l1_off --t_grasp 12 --t_switch 99 --seed 1
%LAUNCH% %COMMON% --condition unknown --l1_off --t_grasp 12 --t_switch 99 --seed 2
REM -- separated: unknown/L1-off --
%LAUNCH% %COMMON% --condition unknown --l1_off --t_grasp 12 --t_switch 17 --seed 0
%LAUNCH% %COMMON% --condition unknown --l1_off --t_grasp 12 --t_switch 17 --seed 1
%LAUNCH% %COMMON% --condition unknown --l1_off --t_grasp 12 --t_switch 17 --seed 2
echo Wave 2 complete.