matFilePath = "E:\Dropbox_transfer\Neuropixels\9153\RESULT0616\UnitMatch.mat";
UMmatfile = load(matFilePath);
[filepath, ~, ~] = fileparts(matFilePath);

save_path = "E:\Dropbox_transfer\Neuropixels\9153\RESULT0616";

disp(save_path)

uniqueIDStruct = clusinfo;

varNames = {'Good_ID', 'KSLabel', 'RecSesID'};

for i = 1:length(varNames)
    varName = varNames{i};
    
    if isfield(uniqueIDStruct, varName)
        varData = uniqueIDStruct.(varName);
        if size(varData, 1) == 1 && size(varData, 2) > 1
            varData = varData';
        elseif size(varData, 1) > 1 && size(varData, 2) > 1
            varData = varData(:);
        end
        
        csvFileName = fullfile(save_path, [varName '.csv']);
        disp(csvFileName)
        writematrix(varData, csvFileName);
        
        fprintf('saved %s to %s\n', varName, csvFileName);
    else
        fprintf('Warning: %s not exist in UniqueIDConversion\n', varName);
    end
end
