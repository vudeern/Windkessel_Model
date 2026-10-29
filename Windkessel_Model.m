%% 2-Element Windkessel Model – Cardiovascular Assignment
%
%  Circuit (from figure):
%    I(t) [current source] || C [capacitor] || R [resistor]
%    All three share Node A (top) and ground (bottom).
%
%  Analogy:
%    I(t)  = cardiac output flow waveform     [mL/s]
%    V(t)  = aortic pressure                  [mmHg]
%    C     = arterial compliance              [mL/mmHg]
%    R     = peripheral resistance            [mmHg·s/mL]
%
%  KCL at Node A:
%    I(t) = I_C + I_R
%    I(t) = C * dV/dt  +  V/R
%
%  ODE:
%    dV/dt = ( I(t) - V/R ) / C
%
%  Numerical method: ode45 (Dormand-Prince RK4/5 adaptive step)

clear; close all; clc;

%% ═══════════════════════════════════════════════════════════════════════════
%  PARAMETERS
% ════════════════════════════════════════════════════════════════════════════
HR      = 60;
T       = 60 / HR;
t_sys   = T / 3;
SV      = 70;
Q_peak  = pi*SV/(2*t_sys);

R       = 1.0;
C_nom   = 1.5;

n_cycles = 10;
dt      = 0.001;
t       = 0 : dt : n_cycles*T - dt;

%% ═══════════════════════════════════════════════════════════════════════════
%  FLOW WAVEFORM  I(t)
% ════════════════════════════════════════════════════════════════════════════
function I = flow_waveform(t, T, t_sys, Q_peak, rf)
    t_dia  = T - t_sys;
    t_mod  = mod(t, T);
    I      = zeros(size(t));

    idx_sys = t_mod < t_sys;
    I(idx_sys) = Q_peak * sin(pi * t_mod(idx_sys) / t_sys);

    if rf > 0
        Q_reg = pi * rf * Q_peak * t_sys / (2 * t_dia);
        idx_dia = t_mod >= t_sys;
        I(idx_dia) = -Q_reg * sin(pi * (t_mod(idx_dia) - t_sys) / t_dia);
    end
end

%% ═══════════════════════════════════════════════════════════════════════════
%  ODE45 SOLVER  – replaces ab2_solve
%  ode45 needs dV/dt as a function of (t, V).
%  Because I(t) is defined on a fixed grid, we interpolate it inside the RHS.
% ════════════════════════════════════════════════════════════════════════════
function V_out = ode45_solve(t_span, C, R, I_vec, t_grid)
    % Interpolate the pre-computed flow vector at any time ode45 requests
    I_interp = @(tq) interp1(t_grid, I_vec, tq, 'linear', 'extrap');

    % RHS of the ODE:  dV/dt = ( I(t) - V/R ) / C
    odefun = @(tq, V) ( I_interp(tq) - V/R ) / C;

    V0   = 80;                          % initial pressure [mmHg]
    opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);

    [~, V_sol] = ode45(odefun, t_span, V0, opts);

    % ode45 returns V at its own adaptive time points → re-sample onto t_span
    % (when t_span is a vector ode45 returns values AT those exact points)
    V_out = V_sol';   % row vector to match original convention
end

%% ═══════════════════════════════════════════════════════════════════════════
%  HELPER: extract last-cycle steady-state metrics
% ════════════════════════════════════════════════════════════════════════════
function [P_sys, P_dia, PP, MAP] = hemodynamics(t, V, T, n_cycles)
    idx   = t >= (n_cycles - 1) * T;
    P_sys = max(V(idx));
    P_dia = min(V(idx));
    PP    = P_sys - P_dia;
    MAP   = mean(V(idx));
end

%% ═══════════════════════════════════════════════════════════════════════════
%  PART 1 – NORMAL FLOW, SHOW I(t)
% ════════════════════════════════════════════════════════════════════════════
I_normal = flow_waveform(t, T, t_sys, Q_peak, 0);

figure('Color','w','Name','Part 1 – Flow Waveform','Position',[50 500 700 300]);
mask2 = t <= 2*T;
plot(t(mask2), I_normal(mask2), 'b-', 'LineWidth', 2);
xline(t_sys,  '--r', 'End systole', 'LabelHorizontalAlignment','left');
xline(T,      '--k', 'End cycle 1');
xline(T+t_sys,'--r', 'LabelHorizontalAlignment','left');
xlabel('Time (s)'); ylabel('Flow I(t)  (mL/s)');
title('Cardiac Flow Waveform – Normal (HR 60 bpm, SV 70 mL)');
legend(sprintf('Q_{peak} = %.0f mL/s', Q_peak), 'Location','northeast');
grid on; box off;
ylim([-20, Q_peak*1.15]);

%% ═══════════════════════════════════════════════════════════════════════════
%  PART 1 – PULSE PRESSURE vs COMPLIANCE  (3 levels)
% ════════════════════════════════════════════════════════════════════════════
C_vals   = [1.5, 2.0, 2.5];
C_labels = {'C = 1.5 (stiff)', 'C = 2.0 (normal)', 'C = 2.5 (compliant)'};
clrs     = [0.85 0.10 0.10; 0.13 0.47 0.71; 0.18 0.63 0.18];

figure('Color','w','Name','Part 1 – Compliance Effect','Position',[50 150 900 420]);
t_cyc = 0 : dt : T - dt;

fprintf('\n── Part 1: Compliance Sensitivity ─────────────────────────\n');
fprintf('  %-24s  Psys  Pdia   PP   MAP\n','Condition');
for k = 1:3
    I_k = flow_waveform(t, T, t_sys, Q_peak, 0);
    V_k = ode45_solve(t, C_vals(k), R, I_k, t);   % <-- ode45 call

    idx_lc = t >= (n_cycles-1)*T;
    V_lc   = V_k(idx_lc);
    V_lc   = V_lc(1:length(t_cyc));

    plot(t_cyc, V_lc, '-', 'Color', clrs(k,:), 'LineWidth', 2, ...
         'DisplayName', C_labels{k}); hold on;

    [Ps, Pd, PP, MA] = hemodynamics(t, V_k, T, n_cycles);
    fprintf('  %-24s  %5.1f %5.1f %5.1f %5.1f\n', C_labels{k}, Ps, Pd, PP, MA);
end
xlabel('Time in cycle (s)'); ylabel('Pressure (mmHg)');
title('Aortic Pressure – Effect of Arterial Compliance');
legend('Location','northeast'); grid on; box off;
fprintf('────────────────────────────────────────────────────────────\n\n');

%% ═══════════════════════════════════════════════════════════════════════════
%  AORTIC REGURGITATION – FLOW WAVEFORMS  I(t)
% ════════════════════════════════════════════════════════════════════════════
RF_vals   = [0, 0.10, 0.30];
RF_labels = {'Normal (RF = 0)', 'Mild AR (RF = 10%)', 'Moderate AR (RF = 30%)'};
clrs_ar   = [0.13 0.47 0.71; 0.85 0.54 0.10; 0.85 0.10 0.10];

figure('Color','w','Name','AR – Flow Waveforms','Position',[100 500 800 320]);
for k = 1:3
    I_k = flow_waveform(t, T, t_sys, Q_peak, RF_vals(k));
    plot(t(mask2), I_k(mask2), '-', 'Color', clrs_ar(k,:), ...
         'LineWidth', 2, 'DisplayName', RF_labels{k}); hold on;
end
yline(0, '--k', 'LineWidth', 0.8);
xlabel('Time (s)'); ylabel('Flow I(t)  (mL/s)');
title('Aortic Flow Waveform – Normal vs Aortic Regurgitation');
legend('Location','northeast'); grid on; box off;

%% ═══════════════════════════════════════════════════════════════════════════
%  AORTIC REGURGITATION – PRESSURE TRACINGS
% ════════════════════════════════════════════════════════════════════════════
figure('Color','w','Name','AR – Pressure Tracings','Position',[100 150 900 420]);

fprintf('── Aortic Regurgitation Sensitivity ────────────────────────\n');
fprintf('  %-26s  Psys  Pdia   PP   MAP\n','Condition');
for k = 1:3
    I_k = flow_waveform(t, T, t_sys, Q_peak, RF_vals(k));
    V_k = ode45_solve(t, C_nom, R, I_k, t);       % <-- ode45 call

    idx_lc = t >= (n_cycles-1)*T;
    V_lc   = V_k(idx_lc);
    V_lc   = V_lc(1:length(t_cyc));

    plot(t_cyc, V_lc, '-', 'Color', clrs_ar(k,:), 'LineWidth', 2, ...
         'DisplayName', RF_labels{k}); hold on;

    [Ps, Pd, PP, MA] = hemodynamics(t, V_k, T, n_cycles);
    fprintf('  %-26s  %5.1f %5.1f %5.1f %5.1f\n', RF_labels{k}, Ps, Pd, PP, MA);
end
fprintf('────────────────────────────────────────────────────────────\n\n');

xlabel('Time in cycle (s)'); ylabel('Pressure (mmHg)');
title('Aortic Pressure – Effect of Aortic Regurgitation (C = 1.5)');
legend('Location','northeast'); grid on; box off;

fprintf('All figures generated.\n');