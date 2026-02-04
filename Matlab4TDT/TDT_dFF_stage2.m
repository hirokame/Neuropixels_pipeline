function dFF = TDT_dFF_stage2(session_name,Stem_Dir,Save_univ_dir0,Save_univ_dir1,Save_univ_dir2,chunky_or_not)
%%   
% clear;
% session_name    = '/Users/ayanomatsushima/Library/CloudStorage/GoogleDrive-ayanom@mit.edu/My Drive/Graybiel_Lab/AM_data/TDT_photometry/Tanks/a00_a00_a9493_a9496-250129-154009/Mouse-250206-131745';
% session_name = ['/Users/ayanomatsushima/Library/CloudStorage/GoogleDrive-ayanom@mit.edu/My Drive/Graybiel_Lab/AM_data/' ...
%          'TDT_photometry/Tanks/a7816_a7818_a7789_a7793_240805-240806-103028/Mouse-240814-102614'];
% Save_univ_dir0 = '/Users/ayanomatsushima/Library/CloudStorage/GoogleDrive-ayanom@mit.edu/My Drive/Graybiel_Lab/AM_data/photometry/Unv_matlab_out_byMs/stage0';
% Save_univ_dir1 = '/Users/ayanomatsushima/Library/CloudStorage/GoogleDrive-ayanom@mit.edu/My Drive/Graybiel_Lab/AM_data/photometry/Unv_matlab_out_byMs/stage1';
% Save_univ_dir2 = '/Users/ayanomatsushima/Library/CloudStorage/GoogleDrive-ayanom@mit.edu/My Drive/Graybiel_Lab/AM_data/photometry/Unv_matlab_out_byMs/stage2';
% figsavedir = '/Users/ayanomatsushima/Library/CloudStorage/GoogleDrive-ayanom@mit.edu/My Drive/Graybiel_Lab/ProgressReports@MIT/250215_afterLM/250218/';
% Stem_Dir = '/Users/ayanomatsushima/Library/CloudStorage/GoogleDrive-ayanom@mit.edu/My Drive/Graybiel_Lab/AM_data/photometry/Unv_matlab_out_byMs';
% chunky_or_not = 0;
      
%% Extract mouseID and sessionDate
close all;
if isempty(session_name);
    dFF = [];
    return;
end
try
    split_dir = split(session_name,'\');
catch
    disp(['TDT_dFF_stage2: cannot split ' session_name])
    return;
end
temp = split_dir{end};
temp = split(temp,'-');
Date2check = string(cell2mat(temp(2)));

if length(split_dir) < 2
    return;
end
mouseIDs = split_dir{end-1};
temp = split(mouseIDs,'-');
temp = split(temp{1},'_');
if length(temp) < 4; 
    warning('TDT_dFF_stage2: dFF_stage2:warining','Temp length is less than 4')
    disp(temp)
    return; 
end
for i = 1:4 % box number
    mouseIDwithDOB(i) = string(replace(temp{i},'k',''));
end
%%

% other parameters
Smoothing_Value = 3;
detrend_sec     = 10;
cutoff          = 10;

% Cut first and last ms
BufferInput   = [0 0]; % in ms

% time range to show (xlim)
show_s   = 5;
lowpass_order = 2; % with Dan 250226

%%
% If exist, load, if not, calculate (just readout TDT file)
temp = split(char(session_name),'\');
check_folder = temp{end};
previous_file = [Save_univ_dir1 '/Session_' check_folder '_UnivRAW_offdemod.mat'];
if string(check_folder) == "Mouse-250219-101858" % except for this
    return;
end
if ~exist(previous_file)
    TDT_demod(session_name,Save_univ_dir0,Save_univ_dir1,1);
    if ~exist(previous_file);
        return;
    end
    fileSize = get_filesize(previous_file);
    disp(['TDT_dFF_stage2: loading Stage1 ; size ' num2str(fileSize*1e-9) 'GB'])
    load(previous_file)
else
    disp(['TDT_dFF_stage2: Stage1 was already calculated for ' check_folder])
    load(previous_file)
    % try
    %     fileSize = get_filesize(previous_file);
    %     disp(['TDT_dFF_stage2: loading Stage1 ; size ' num2str(fileSize*1e-9) 'GB'])
    %     load(previous_file)
    % catch
    %     disp(['TDT_dFF_stage2: Stage1 was already calculated but cannot load ' check_folder])
    %     TDT_demod(session_name,Save_univ_dir0,Save_univ_dir1,1);
    % end
end
if ~exist('orig_Fs','var');
    TDT_demod(session_name,Save_univ_dir0,Save_univ_dir1,1);
end
if ~exist(previous_file);
    return;
end
output_file = [Save_univ_dir2 '/Session_' check_folder  '_dFF.mat'];
% if exist(output_file)
%     disp(['TDT_dFF_stage2: Stage2 was already calculated for ' check_folder])
%     return;
% end
if ~exist(previous_file);
    return;
end
disp(['TDT_dFF_stage2: Post-LP1 and Calculating dFF for ' check_folder])
%% Find indexes to include
% check if the cutting ratio is correct
whole_session = size(handles.Fi1rOffDmd405_1,1); % in second
Start_ix = round(handles.Fs*BufferInput(1)/1e3);
Start_ix = 1;
End_ix   = round(whole_session-handles.Fs*BufferInput(2)/1e3);

% if this is too low, cut too much
ratio_analyzed = size(Start_ix:End_ix,2) / whole_session
%% Modify trial starts timestamps so that whole session shoud be 1

% % default, common parameters calculated in TDTdemod
% fieldName  = 'x405A';
% whole_session_sampleNs = numel(handles.data.streams.(fieldName).data(1,:));
% orig_Fs                = handles.data.streams.(fieldName).fs;
%%
fields = fieldnames(handles);
for i = 1:4
    Ts = handles.Ts{i};
    Ts = Ts/max(Ts); % set the whole recording duration to 1
    Ts = Ts(Start_ix:End_ix);
    
    fieldName    = ['Fi' num2str(i) 'r']; % Box number = sensor number
    outfieldName4051_orig = [fieldName 'OffDmd405_1'];
    outfieldName4651_orig = [fieldName 'OffDmd465_1'];
    outfieldName4052_orig = [fieldName 'OffDmd405_2'];
    outfieldName5602_orig = [fieldName 'OffDmd560_2'];
    % Extract data from samples between the specified start and end times:
    Signals(i).orig_4051 = handles.(outfieldName4051_orig)(Start_ix:End_ix);
    Signals(i).orig_4651 = handles.(outfieldName4651_orig)(Start_ix:End_ix);
    Signals(i).orig_4052 = handles.(outfieldName4052_orig)(Start_ix:End_ix);
    Signals(i).orig_5602 = handles.(outfieldName5602_orig)(Start_ix:End_ix);
    
    ch_num = 0;
    for channel_color = [4051 4651 4052 5602]
        ch_num = ch_num + 1;
        fieldnameOrig = ['orig_'   num2str(channel_color)];
        fieldnameFilt = ['Filt'     num2str(channel_color)];

        Signals(i).(fieldnameFilt) = lowpass_butter_sos_gain1(Signals(i).(fieldnameOrig),...
                                        handles.Fs,lowpass_order,1);
        legend_txt = 'LP1Hz';
           
        % plot
        figure(i);set(gcf,'position',[1    69   667   797])
        subplot(4,2,2*(ch_num-1)+1);hold on; title([num2str(channel_color) ' demodulated and firstLP ' num2str(cutoff) ' in qd Box' num2str(i) ' for ' num2str(show_s) 's from 30min']);
        plot(Ts,Signals(i).(fieldnameOrig));
        plot(Ts,Signals(i).(fieldnameFilt));
        
        xlim(1/2+[0 show_s/3600]); % assume session was 1hr
        xlim_range = xlim;
        xtick_loci = linspace(xlim_range(1),xlim_range(2),show_s+1);
        set(gca,'xtick',xtick_loci,'xticklabel',0:show_s);
        xlabel('Time(s)');ylabel('F');
        legend('Before Filt',legend_txt)
    end
%% calculate dF/F using different controls
    for pair_n = 1:3
        subplot(3,2,2*pair_n);hold on;
        switch pair_n
            case 1
                Signal  = Signals(i).Filt4651;
                Control = Signals(i).Filt4051;
                title({['dF/F 465(sensor1) with 405(sensor1) control demodulated'],['and firstLP ' num2str(cutoff) ' in qd in Box' num2str(i)  ' for ' num2str(show_s) 's from 30min']})
            case 2
                Signal  = Signals(i).Filt5602;
                Control = Signals(i).Filt4051;
                title({['dF/F 560(sensor2) with 405(sensor1) control demodulated'],['and firstLP ' num2str(cutoff) ' in qd  in Box' num2str(i)  ' for ' num2str(show_s) 's from 30min']})
            case 3
                Signal  = Signals(i).Filt5602;
                Control = Signals(i).Filt4052;
                title({['dF/F 560(sensor2) with 405(sensor2) control demodulated'],['and firstLP ' num2str(cutoff) ' in qd  in Box' num2str(i)  ' for ' num2str(show_s) 's from 30min']})
        end
        Signal  = fillmissing(Signal,'linear');
        Control = fillmissing(Control,'linear');
        global F405raw F465raw dFF_raw
        F405raw = Signals(i).orig_4051;
        F465raw = Signals(i).orig_4651;
        dFF_raw = DeltaF(Signal,Control,Smoothing_Value);
        hann_size = round(detrend_sec*handles.Fs);
        hann_size = hann_size + ~mod(hann_size,2);
        dFF_out{i}{pair_n} = dFF_raw - conv(dFF_raw,hann(hann_size)/sum(hann(hann_size)),'same');
        plot(Ts,dFF_out{i}{pair_n});hold on
        
        % save data for session
        dFFOut.box(i).pair(pair_n).data = dFF_out{i}{pair_n};
        dFFOut.box(i).pair(pair_n).Fs = handles.Fs;
        dFFOut.box(i).pair(pair_n).Ts = Ts;

        % save data for each box
        dFF.pair(pair_n).data = dFF_out{i}{pair_n};
        dFF.pair(pair_n).Fs = handles.Fs;
        dFF.pair(pair_n).Ts = Ts;
        
        % save data for session
        % trialstarts are recorded in the same frequency as x405A
        dFFOut.box(i).trialstarts = orig_Fs* handles.TrialStarts(i).time/whole_session_sampleNs; 
        % save data for each box
        dFF.trialstarts = orig_Fs* handles.TrialStarts(i).time/whole_session_sampleNs;
%         show = [min(dFF.trialstarts) max(dFF.trialstarts)]
%         show = [min(Ts) max(Ts)]
    end
    for sub_k = 1:3
        subplot(3,2,2*sub_k);
        ylabel('dF/F');
        xlim(1/2+[0 show_s/3600]); % assume session was 1hr
        xlim_range = xlim;
        xtick_loci = linspace(xlim_range(1),xlim_range(2),show_s+1);
        set(gca,'xtick',xtick_loci,'xticklabel',0:show_s);
        xlabel('Time(s)');
        legend(legend_txt)
    end
    
    if ~isempty(char(mouseIDwithDOB(i)))
        %% save data
        dir_read = [Stem_Dir '\' char(mouseIDwithDOB(i)) '\' char(Date2check)];
        if ~exist(dir_read); mkdir(dir_read); end
        save([dir_read '\' char(mouseIDwithDOB(i)) '_' char(Date2check)  '_dFF.mat'],'dFF')
        %% save figures
        saveas(i,[dir_read '\' char(mouseIDwithDOB(i)) '_' char(Date2check)  '_dFF'],'fig')
        saveas(i,[dir_read '\' char(mouseIDwithDOB(i)) '_' char(Date2check)  '_dFF'],'png')
    end
end
%% save data
% all boxes
disp(['TDT_dFF_stage2: saving dFF for ' check_folder])
save(output_file,'dFFOut')

