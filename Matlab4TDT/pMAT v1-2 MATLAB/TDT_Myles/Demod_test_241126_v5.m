clear 
close all;
add_path_parent1 = '/Users/ayanomatsushima/Documents/MATLAB/codes';
disp(['Adding path under ' add_path_parent1])
addpath(genpath(add_path_parent1))
disp(['Added path under ' add_path_parent1])
%%
% a7816_a7818_a00_a00-240628-101850
data = 'Mouse-240703-094320'; % First Pav
% data = 'Mouse-240731-101519'; % First PR
% data = 'Mouse-240801-101054'; % FR1
% data = 'Mouse-240813-104820'; % 5th PR

% a8851_a8570_a8569_a8756-240925-113656
% data = 'Mouse-240930-103818'; % First Pav
% data = 'Mouse-241010-112549'; % 6th Pav
% data = 'Mouse-241007-111909'; % First PR
% data = 'Mouse-241017-113826'; % 3rd PR

% a8757_a8564_a8568_a8567-240925-142146
% data = 'Mouse-240930-123620'; % First Pav
% data = 'Mouse-241010-134956'; % 6th Pav
data = 'Mouse-241007-125006'; % First PR
% data = 'Mouse-241017-130352'; % 3rd PR

load_type = '_UnivRAW_DeltaF_updown.mat'; % only use handles.Ch465 etc, so no need to use files below
% load_type = '_UnivRAW_DeltaF_noDownnoFiltnoHighpass.mat';
% load_type = '_UnivRAW_DeltaF_noDownnoFiltDetrend.mat';
load(['/Users/ayanomatsushima/Library/CloudStorage/GoogleDrive-ayanom@mit.edu/My Drive/Graybiel_Lab/AM_data/photometry/Univ_matlab_out/stage1/Session#' data load_type]);

one_session_data = handles.data.streams;
save(['/Users/ayanomatsushima/Downloads/Demodulation/d' data '.mat'],'one_session_data');

%%
F1_1 = handles.data.streams.Fi1r.data(1,:); % sensor 1
F2_1 = handles.data.streams.Fi1r.data(2,:); % sensor 2
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

onlinedemod405 = handles.Ch405A;
onlinedemod465 = handles.Ch465A;
onlinedemod560 = handles.Ch560B;

%% Truncate

BufferInput = {'20000','2000'}; %%%%%%%%%%%%%%%
StartBuffer = str2double(BufferInput{1});
EndBuffer   = str2double(BufferInput{2});
EndTime     = length(handles.data.streams.x405A.data)-EndBuffer;

ydemod405_1 = ydemod405_1(StartBuffer:EndTime);
ydemod405_2 = ydemod405_2(StartBuffer:EndTime);
ydemod465_1 = ydemod465_1(StartBuffer:EndTime);
ydemod465_2 = ydemod465_2(StartBuffer:EndTime);
ydemod560_1 = ydemod560_1(StartBuffer:EndTime);
ydemod560_2 = ydemod560_2(StartBuffer:EndTime);

% size(onlinedemod405)
% size(ydemod405_1)
%% Sensor 1 for 465, sensor 2 for 560
basic_figures_plot(ydemod405_1,ydemod405_2,ydemod465_1,ydemod465_2,ydemod560_1,ydemod560_2)

%% compare online and offline
show_equal = [isequal(onlinedemod405,ydemod405_1)
              isequal(onlinedemod465,ydemod465_1)
              isequal(onlinedemod560,ydemod560_2)]
figure;
subplot(3,1,1);plot(onlinedemod405,ydemod405_1)
subplot(3,1,2);plot(onlinedemod465,ydemod465_1)
subplot(3,1,3);plot(onlinedemod560,ydemod560_2)

%% Parameters set
detrend_sec  = 10; % sec
portion      = .2; % 20% for xcov
blowup_win_s = 2;
blowup_win_s = 300;
mag_factor   = 1e3;
style        = '-';
fs           = Fs;
FiSize       = size(handles.data.streams.Fi1r.data,2);
TsSize       = size(ydemod405_1,1);
Ts           = linspace(0,FiSize/Fs,TsSize);
Smoothing_on = 0;
Lowpass_Fq   = 2;
Iakovos_base = 1;
highpass_yn  = 0;

if Smoothing_on 
    Smoothing_Value = round(Fs/Lowpass_Fq);
else
    Smoothing_Value = 3;
end
%%
for plot_signals_n = 1:4
    switch plot_signals_n
        case 1
            Ch405 = ydemod405_1';
            Ch465 = ydemod465_1';
            Ch560 = ydemod560_1';
            TTs = Ts;
        case 2
            Ch405 = ydemod405_2';
            Ch465 = ydemod465_2';
            Ch560 = ydemod560_2';
            TTs = Ts;
        case 3
            Ch405 = ydemod405_1';
            Ch465 = ydemod465_2';
            Ch560 = ydemod560_2';
            TTs = Ts;
        case 4
            Ch405 = onlinedemod405;
            Ch465 = onlinedemod465;
            Ch560 = onlinedemod560;
            TTs = handles.Ts;
    end
    draw_step1(Ch405,Ch465,Ch560,detrend_sec,fs,TTs,portion,blowup_win_s,mag_factor,style)
    switch plot_signals_n
        case 1
            title('Sensor 1')
        case 2
            title('Sensor 2')
        case 3
            title('sensor 1 (405) and 2')
        case 4
            title('Online demod')
    end
    % keyboard;
end
%% different controls

for pair_n = 1:6
    switch pair_n
        case 1
            Signal  = ydemod465_1;
            Control = ydemod405_1;
        case 2
            Signal  = ydemod560_2;
            Control = ydemod405_1;
        case 3
            Signal  = ydemod560_2;
            Control = ydemod405_2;
        case 4
            Signal  = ydemod560_2;
            Control = ydemod465_2;
        case 5
            Signal  = onlinedemod465;
            Control = onlinedemod405;
        case 6
            Signal  = onlinedemod560;
            Control = onlinedemod405;
    end
    dFF_raw = DeltaF(Signal,Control,Smoothing_Value);

    if highpass_yn
        fcutlow = 0.05; 
        dFF_out = highpass(dFF_raw,fcutlow,fs);
    elseif Iakovos_base
        dFF_out = dFF_raw - movmedian(dFF_raw,detrend_sec*fs);
    else 
        dFF_out = dFF_raw;
    end
    dFF_final{pair_n} = dFF_out;
end
%%
pair_mat = [1 2
            1 3
            1 4
            5 6];
pair_mat_names = {'465_to1 and 560_2to1',...
                  '465_to1 and 560_2to2',...
                  '465_to1 and 560_2to2G',...
                  'online465 and online560'};
for pair_nn = 1:size(pair_mat,1)
    dFF465 = dFF_final{pair_mat(pair_nn,1)};
    dFF560 = dFF_final{pair_mat(pair_nn,2)};
    % temporal correlation
    maxlag = round(length(dFF_raw)*portion);
    lags = [-maxlag:maxlag]*median(diff(Ts));
    zoomlag = round(maxlag/mag_factor);
    zeropt = find(lags==0);
    xcov_smoothGR = xcov(dFF465,dFF560,maxlag,'normalized');
    plot_data(lags,xcov_smoothGR,...
             {'465 and 560'},'Delay(s)','normalized correlation',...
              ['Temporal correlation dF/F ' pair_mat_names(pair_nn)],[2740   33   400   422],style)
    plot([0 0],ylim,'k--')
    [v ix] = max(abs(xcov_smoothGR));
    max_cor_delay = lags(ix);
    text(max_cor_delay,max(ylim)*.8,['delay = ' num2str(max_cor_delay)])

    % same plot blow up
    plot_data(lags((-zoomlag:zoomlag) + zeropt),xcov_smoothGR((-zoomlag:zoomlag) + zeropt),...
             {'465 and 560'},'Delay(s)','normalized correlation',...
              ['Temporal correlation dF/F ' pair_mat_names(pair_nn)],[3140   33   400   422],style)
    plot([0 0],ylim,'k--')
    text(max_cor_delay,max(ylim)*.8,['delay = ' num2str(max_cor_delay)])
end
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
% nsignals = 3;
% bandwidth_F = [10 15];
% 
% [c_Mag, Ref_F, PSDs, cache, c_Raw] = contdemodulate(c_Raw, 'nsignals', nsignals, 'bandwidth_F',bandwidth_F)
%%
% handles.TDTFold  = '/Users/ayanomatsushima/Library/CloudStorage/GoogleDrive-ayanom@mit.edu/My Drive/Graybiel_Lab/AM_data/TDT_photometry/Tanks/a8757_a8564_a8568_a8567-240925-142146/Mouse-241007-125006';
% cd(handles.TDTFold);
% [Tank,Block,~]=fileparts(cd);
% handles.data=TDTbin2mat([Tank,'\',Block]);

%%
% dem405_1 = demod(F1_1,frqs_405,handles.Fs);
% dem465_1 = demod(F1_1,frqs_465,handles.Fs);
% % dem560_1 = demod(F1_1,frqs_560,handles.Fs);
% dem405_2 = demod(F2_1,frqs_405,handles.Fs);
% dem465_2 = demod(F2_1,frqs_465,handles.Fs);
% % dem560_2 = demod(F2_1,frqs_560,handles.Fs);
% 
% figure;
% plot([dem405_1' dem405_2'])
% figure;
% plot([dem465_1' dem465_2'])