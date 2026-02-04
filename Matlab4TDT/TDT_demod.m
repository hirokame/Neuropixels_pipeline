function TDT_demod(session_name,Save_univ_dir0,Save_univ_dir1,reload)

% clear; close all;
% session_name = ['/Users/ayanomatsushima/Library/CloudStorage/GoogleDrive-ayanom@mit.edu/My Drive/Graybiel_Lab/AM_data/' ...
%          'TDT_photometry/Tanks/a7816_a7818_a7789_a7793_240805-240806-103028/Mouse-240814-102614'];
% user_name      = '/Users/ayanomatsushima/Library';
% Save_univ_dir0 = [user_name '/CloudStorage/GoogleDrive-ayanom@mit.edu/My Drive/Graybiel_Lab/AM_data/photometry/Unv_matlab_out_byMs/stage0'];
% Save_univ_dir1  = [user_name '/CloudStorage/GoogleDrive-ayanom@mit.edu/My Drive/Graybiel_Lab/AM_data/photometry/Unv_matlab_out_byMs/stage1'];
%% check if file already exists
fclose('all');
folder_name_split = split(session_name,'\');
check_folder = folder_name_split{end};

file_stage0 = [Save_univ_dir0 '/Session_' check_folder '_UnivRAW_stage0.mat'];
if exist(file_stage0)
    disp(['TDT_demod: Stage0 was already calculated for ' check_folder]) 
    try
        load(file_stage0,'handles')
    catch
        handles = TDT_read_stage0(session_name,Save_univ_dir0);
        load(file_stage0,'handles')
    end
else
    disp(['TDT_demod: Session_' check_folder '_UnivRAW_stage0.mat does not exist!']);
    handles = TDT_read_stage0(session_name,Save_univ_dir0);
end

output_file = [Save_univ_dir1 '/Session_' check_folder '_UnivRAW_offdemod.mat'];
if exist(output_file) & ~reload
    disp(['TDT_demod: Stage1 was already calculated for ' check_folder]) 
    return;
end
if ~exist(file_stage0)
    return;
end
disp(['TDT_demod: Demodulating session ' check_folder]) 
%% key parameters
frqs_405 = 210;
frqs_465 = 330;
frqs_560 = 530;
%%
folder_name_split = split(session_name,'\');
%% Make dir and extract box names
date = char(datetime('today','Format','yyyy-MM-dd'));
Save_dir  = [Save_univ_dir1 '\' date '\'];
if ~exist(Save_univ_dir1); mkdir(Save_univ_dir1); end
if ~exist(Save_dir);  mkdir(Save_dir); end

[Box_name check_folder] = extract_Box_name(session_name);

if length(Box_name)~=4
    warning('Box number is not equal to 4!')
    disp(Box_name)
    if length(Box_name)<4
        return;
    end
end

%% Initialize handles

handles.Ch490=[];handles.Ch405=[];
handles.Fs=[];handles.Ts=[];handles.fields=[];
handles.Beh=[];handles.epoclist=[];handles.fields=[];

handles.Ch465A=[];handles.Ch405A=[];handles.Ch560B=[];
handles.Ch465C=[];handles.Ch405C=[];handles.Ch560D=[];
handles.Ch465E=[];handles.Ch405E=[];handles.Ch560F=[];
handles.Ch465G=[];handles.Ch405G=[];handles.Ch560H=[];

handles.Tr_1_on=[]; handles.Tr_1_off=[];
handles.Tr_2_on=[]; handles.Tr_2_off=[]; 
handles.Tr_3_on=[]; handles.Tr_3_off=[];  
handles.Tr_4_on=[]; handles.Tr_4_off=[];
handles.Op_1_on=[]; handles.Op_1_off=[];  
handles.Op_2_on=[]; handles.Op_2_off=[]; 
handles.Op_3_on=[]; handles.Op_3_off=[];  
handles.Op_4_on=[]; handles.Op_4_off=[];
%% Read channels
fields     = handles.streamlist;

% Using a for loop to read the values of each field
for i = 1:length(fields)
    fieldName    = fields{i};
    if fieldName(1) ~= 'F'
        outfieldName = ['Ch' fieldName(2:end)];
        % Extract data from samples between the specified start and end times:
        handles.(outfieldName) = handles.data.streams.(fieldName).data;
        fs = handles.data.streams.(fieldName).fs;
    else
        outfieldName405_1 = [fieldName 'OffDmd405_1']; % field name has box number, e.g., Fi1r = box1
        outfieldName405_2 = [fieldName 'OffDmd405_2'];
        outfieldName465_1 = [fieldName 'OffDmd465_1'];
        outfieldName465_2 = [fieldName 'OffDmd465_2'];
        outfieldName560_1 = [fieldName 'OffDmd560_1'];
        outfieldName560_2 = [fieldName 'OffDmd560_2'];
        F1 = handles.data.streams.(fieldName).data(1,:); % sensor 1
        F2 = handles.data.streams.(fieldName).data(2,:); % sensor 2
        Fs = handles.data.streams.(fieldName).fs;

        [handles.(outfieldName405_1), cfg, figs] = quickdemod(F1,Fs,frqs_405);
        [handles.(outfieldName405_2), cfg, figs] = quickdemod(F2,Fs,frqs_405);
        [handles.(outfieldName465_1), cfg, figs] = quickdemod(F1,Fs,frqs_465);
        [handles.(outfieldName465_2), cfg, figs] = quickdemod(F2,Fs,frqs_465);
        [handles.(outfieldName560_1), cfg, figs] = quickdemod(F1,Fs,frqs_560);
        [handles.(outfieldName560_2), cfg, figs] = quickdemod(F2,Fs,frqs_560);
        handles.Fs = cfg.downsample_fs;
        handles.Ts{str2num(fieldName(3))} = [0:(length(handles.(outfieldName560_2))-1)]/handles.Fs;
        a = 0;
    end
end
%% Extract events - behavior data
handles.fields = fieldnames(handles.data.epocs);
handles.MasterArray=[];

for i=1:numel(handles.fields)
    epoclist=cell(numel(handles.data.epocs.(handles.fields{i}).data),1);
    epoclist(:)=handles.fields(i);
    handles.data.epocs.(handles.fields{i}).data=epoclist;
    % Using num2str with cellfun converts each element of the onset and offset arrays from numbers into strings.
    onset =cellfun(@num2str, num2cell(handles.data.epocs.(handles.fields{i}).onset),'un',0);
    offset=cellfun(@num2str, num2cell(handles.data.epocs.(handles.fields{i}).offset),'un',0);
    % Create a new matrix for current epoc:
    BehData.(handles.fields{i})=[epoclist onset offset];
    handles.MasterArray= [handles.MasterArray; BehData.(handles.fields{i})];
end
names = {'Tr','Op'};
for Tr_or_Op = 1:2
    for inside_name_nn = 1:4
        fieldname = [cell2mat(names(Tr_or_Op)) '_' num2str(inside_name_nn)];
        if isfield(handles.data.epocs,fieldname)
            field_on = [cell2mat(names(Tr_or_Op)) '_' num2str(inside_name_nn) '_on'];
            field_off= [cell2mat(names(Tr_or_Op)) '_' num2str(inside_name_nn) '_off'];
            handles.(field_on)  = handles.data.epocs.(fieldname).onset;
            handles.(field_off) = handles.data.epocs.(fieldname).offset;
        end
    end
end
handles.TrialStarts(1).time = handles.Tr_1_on;
handles.TrialStarts(2).time = handles.Tr_2_on;
handles.TrialStarts(3).time = handles.Tr_3_on;
handles.TrialStarts(4).time = handles.Tr_4_on;
%% intensity etc setting
if ~isempty(find(ismember(fieldnames(handles.data.scalars),'Fi1i')))
    DrFreq_level_offset{1} = handles.data.scalars.Fi1i.data;
end
if ~isempty(find(ismember(fieldnames(handles.data.scalars),'Fi2i')))
    DrFreq_level_offset{2} = handles.data.scalars.Fi2i.data;
end
if ~isempty(find(ismember(fieldnames(handles.data.scalars),'Fi3i')))
    DrFreq_level_offset{3} = handles.data.scalars.Fi3i.data;
end
if ~isempty(find(ismember(fieldnames(handles.data.scalars),'Fi4i')))
    DrFreq_level_offset{4} = handles.data.scalars.Fi4i.data;
else
    DrFreq_level_offset{4} = [];
end

for fiber_n = 1:4
    if isempty(DrFreq_level_offset{fiber_n}); continue; end
    for color_number = 1:3
        handles.settings(fiber_n).dvfreq(color_number) = DrFreq_level_offset{fiber_n}(2+4*(color_number-1));
        handles.settings(fiber_n).levels(color_number) = DrFreq_level_offset{fiber_n}(3+4*(color_number-1));
        handles.settings(fiber_n).offsets(color_number)= DrFreq_level_offset{fiber_n}(4+4*(color_number-1));
    end
end
% keyboard;
%%
whos handles
%%
fieldName  = 'x405A';
whole_session_sampleNs = numel(handles.data.streams.(fieldName).data(1,:));
orig_Fs                = handles.data.streams.(fieldName).fs;

fields = fieldnames(handles);
toRemove = fields(startsWith(fields, 'Ch'));
handles = rmfield(handles,toRemove);
handles = rmfield(handles,'data');
%%
whos handles
%%
fclose('all');
disp('TDT_demod : saving handles and other two parameters')
try
    save(output_file,'handles','whole_session_sampleNs','orig_Fs','-v7.3')
catch
    delete(output_file)
    save(output_file,'handles','whole_session_sampleNs','orig_Fs','-v7.3')
end
output_file2 = [Save_dir  '/Session_' check_folder '_UnivRAW_offdemod.mat'];
try
    save(output_file2,'handles','whole_session_sampleNs','orig_Fs','-v7.3')
catch
    delete(output_file2)
    save(output_file2,'handles','whole_session_sampleNs','orig_Fs','-v7.3')
end