clear 
close all;

data = 'Mouse-241007-125006'; % First PR

load_type = '_UnivRAW_DeltaF_updown.mat'; % only use handles.Ch465 etc, so no need to use files below
% load_type = '_UnivRAW_DeltaF_noDownnoFiltnoHighpass.mat';
% load_type = '_UnivRAW_DeltaF_noDownnoFiltDetrend.mat';
load(['/Users/ayanomatsushima/Library/CloudStorage/GoogleDrive-ayanom@mit.edu/My Drive/Graybiel_Lab/AM_data/photometry/Univ_matlab_out/stage1/Session#' data load_type]);

one_session_data = handles.data.streams;
save('/Users/ayanomatsushima/Downloads/Demodulation/data_Mouse-241007-125006.mat','one_session_data');

%%
F1_1 = handles.data.streams.Fi1r.data(1,:);
F2_1 = handles.data.streams.Fi1r.data(2,:);
Fs   = handles.data.streams.Fi1r.fs;

frqs_405 = 210;
frqs_465 = 330;
frqs_560 = 530;

[ydemod405_1, cfg, figs] = quickdemod(F1_1,Fs,frqs_405);
[ydemod405_2, cfg, figs] = quickdemod(F2_1,Fs,frqs_405);
[ydemod465_1, cfg, figs] = quickdemod(F1_1,Fs,frqs_465);
[ydemod465_2, cfg, figs] = quickdemod(F2_1,Fs,frqs_465);
[ydemod560_1, cfg, figs] = quickdemod(F1_1,Fs,frqs_560);
[ydemod560_2, cfg, figs] = quickdemod(F2_1,Fs,frqs_560);

figure;
subpot(211); hold on;
plot([ydemod405_1 ydemod405_2 ydemod465_1 ydemod465_2 ydemod560_1 ydemod560_2]);
xlabel('Time(s)');ylabel('Demodulated signals offline')
legend('405 in sensor1','405 in sensor2','465 in sensor1','465 in sensor2','560 in sensor1','560 in sensor2')
subpot(212); hold on;
plot([ydemod405_1 ydemod405_2 ydemod465_1 ydemod465_2 ydemod560_1 ydemod560_2]);
xlabel('Time(s)');ylabel('Demodulated signals offline')
legend('405 in sensor1','405 in sensor2','465 in sensor1','465 in sensor2','560 in sensor1','560 in sensor2')

%%

% [ydemod1, cfg, figs] = quickdemod(F1_1,handles.Fs,frqs_405);
% [ydemod2, cfg, figs] = quickdemod(F2_1,handles.Fs,frqs_405);
% [ydemod3, cfg, figs] = quickdemod(handles.data.streams.Fi1r.data,handles.Fs,frqs_405);

%%
% figure;
% plot(ydemod1); 
% hold on;
% plot(ydemod2)
% 
% figure;
% plot(ydemod3); 
%%
% c_Raw: cont struct with the following named channels:
%           'Det1': Raw detector signal to be demodulated 
%          'Ref1X': carrier sinusoid for signal 1
%          'Ref1Y': 90-degrees out of phase carrier sinusoid for signal 1
%          'Ref2X': carrier sinusoid for signal 2
%          'Ref2Y': 90-degrees out of phase carrier sinusoid for signal 2

% T = 1; % Duration in seconds
% 
% % Time vector
% t = 1:length(handles.Ts);
% 
% % Sinusoidal signal
% RefX405 = sin(2 * pi * t * frqs_405/Fs);
% RefX465 = sin(2 * pi * t * frqs_465/Fs);
% RefX560 = sin(2 * pi * t * frqs_560/Fs);
% RefY405 = cos(2 * pi * t * frqs_405/Fs);
% RefY465 = cos(2 * pi * t * frqs_465/Fs);
% RefY560 = cos(2 * pi * t * frqs_560/Fs);
% 
% % xx = 1:100;
% % figure; plot([RefX405(xx)' RefX465(xx)' RefX560(xx)'])
% % figure; plot([RefX405(xx)' RefY405(xx)'])
% 
% c_Raw.Det1  = F1_1;
% c_Raw.Ref1X = RefX405;
% c_Raw.Ref1Y = RefY405;
% c_Raw.Ref2X = RefX465;
% c_Raw.Ref2Y = RefY465;
% c_Raw.Ref3X = RefX560;
% c_Raw.Ref3Y = RefY560;

% REQUIRED PARAMETERS:
%       'nsignals': How many signals are we demodulating? (no default)
%    'bandwidth_F': Low-pass transition band, in Hz, for frequencies of 
%                   interest in original signal. Mx2, one row per signal. 
%                   If only one row, use same filter for each signal.
%                   (no default, try [10 15])
nsignals = 3;
bandwidth_F = [10 15];

[c_Mag, Ref_F, PSDs, cache, c_Raw] = contdemodulate(c_Raw, 'nsignals', nsignals, 'bandwidth_F',bandwidth_F)
%%
% handles.TDTFold  = '/Users/ayanomatsushima/Library/CloudStorage/GoogleDrive-ayanom@mit.edu/My Drive/Graybiel_Lab/AM_data/TDT_photometry/Tanks/a8757_a8564_a8568_a8567-240925-142146/Mouse-241007-125006';
% cd(handles.TDTFold);
% [Tank,Block,~]=fileparts(cd);
% handles.data=TDTbin2mat([Tank,'\',Block]);

%%
dem405_1 = demod(F1_1,frqs_405,handles.Fs);
dem465_1 = demod(F1_1,frqs_465,handles.Fs);
% dem560_1 = demod(F1_1,frqs_560,handles.Fs);
dem405_2 = demod(F2_1,frqs_405,handles.Fs);
dem465_2 = demod(F2_1,frqs_465,handles.Fs);
% dem560_2 = demod(F2_1,frqs_560,handles.Fs);

figure;
plot([dem405_1' dem405_2'])
figure;
plot([dem465_1' dem465_2'])