function [Box_name check_folder] = extract_Box_name(session_name)
    Box_name = []; 
    check_folder = []; 
    temp = split(char(session_name),'\');
    check_folder = temp{end};
    Tank_name    = temp{end-1};
    temp_info    = split(Tank_name,'_');
    temp_info2   = split(temp_info{end},'-');
    for num = 1:length(temp_info)-1
        Box_name(num) = string(temp_info{num});
    end
    Box_name(length(temp_info)) = string(temp_info2{1});