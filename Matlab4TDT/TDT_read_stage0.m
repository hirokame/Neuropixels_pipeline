function [handles Box_name] =  TDT_read_stage0(session_name,Save_univ_dir)

% clear; close all;
% session_name = ['/Users/ayanomatsushima/Library/CloudStorage/GoogleDrive-ayanom@mit.edu/My Drive/Graybiel_Lab/AM_data/' ...
%          'TDT_photometry/Tanks/a7816_a7818_a7789_a7793_240805-240806-103028/Mouse-240814-102614'];
% user_name      = '/Users/ayanomatsushima/Library';
% Save_univ_dir  = [user_name '/CloudStorage/GoogleDrive-ayanom@mit.edu/My Drive/Graybiel_Lab/AM_data/photometry/Unv_matlab_out_byMs/stage1'];
%% check if file already exists
folder_name_split = split(session_name,'\');
check_folder = folder_name_split{end};
fclose('all');

output_file = [Save_univ_dir '/Session_' check_folder '_UnivRAW_stage0.mat'];
if exist(output_file)
    disp(['TDT_read_stage0: Stage0 was already calculated for ' check_folder]) 
    try
        load(output_file);
        return;
    catch
    end
end
%%
handles.TDTFold = session_name;
cd(handles.TDTFold);
[Tank,Block,~] = fileparts(cd);

FileCheck=dir('*.tev');
filenames = {FileCheck.name};  % Extract the names
temp = filenames(~contains(filenames, '('));  % Exclude names with '('
FileCheck = dir(cell2mat(temp));

if isempty(FileCheck)
    msgbox("No .tev files found. Please select the directory containing your recording data", 'Error','warn');
    return;
else
    disp(['TDT_read_stage0: Processing Block: ', Tank,'\',Block]);
    handles.data = TDTbin2mat([Tank,'\',Block]);
    handles.streamlist = fieldnames(handles.data.streams);
end
% keyboard;
save(output_file,'handles')