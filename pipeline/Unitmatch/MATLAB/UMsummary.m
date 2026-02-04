filefolder =  "E:\Dropbox_transfer\Neuropixels\9153\RESULT0616";

UniqueID      = readmatrix(fullfile(filefolder, 'UniqueID.csv'));
OriginalClusID = readmatrix(fullfile(filefolder, 'OriginalClusID.csv'));
recsesAll     = readmatrix(fullfile(filefolder, 'recsesAll.csv'));
GoodID        = readmatrix(fullfile(filefolder, 'GoodID.csv'));

assert(length(UniqueID) == length(OriginalClusID) && ...
       length(UniqueID) == length(recsesAll) && ...
       length(UniqueID) == length(GoodID), 'CSV not same length');

% only keep the good units
mask = (GoodID == 1);
UniqueID = UniqueID(mask);
OriginalClusID = OriginalClusID(mask);
recsesAll = recsesAll(mask);

[uniqueIDs, ~, idx] = unique(UniqueID);
results = cell(length(uniqueIDs), 5); 

for i = 1:length(uniqueIDs)
    current_mask = (idx == i);
    current_recses = recsesAll(current_mask);
    current_clusID = OriginalClusID(current_mask);
    
    pairs = arrayfun(@(r,c) sprintf('(%d, %d)', r, c), ...
                     current_recses, current_clusID, 'UniformOutput', false);
    pairs_str = ['[' strjoin(pairs, ', ') ']']; 
    session_list = unique(current_recses);
    session_coverage = length(session_list);
    session_list_str = ['[' strjoin(arrayfun(@num2str, session_list, 'UniformOutput', false), ', ') ']'];
    
    results{i, 1} = uniqueIDs(i);  % UniqueID
    results{i, 2} = pairs_str;     % [(recses, OriginalClusID)] pairs
    results{i, 3} = session_coverage; % Session Coverage
    results{i, 4} = session_list_str; % Session List
    results{i, 5} = sum(current_mask); % Number of units
end

result_table = cell2table(results, ...
    'VariableNames', {'UniqueID', 'RecSes_ClusID_Pairs', 'Session_Coverage', 'Session_List', 'Number_of_Units'});
result_table = sortrows(result_table, {'Session_Coverage', 'Number_of_Units'}, {'descend', 'descend'});


writetable(result_table, fullfile(filefolder, 'UnitMatch_Summary_Sorted.csv'));

disp('Finished processing and saved to UnitMatch_Summary_Sorted.csv');