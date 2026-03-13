tank_dirs = {
    'Y:\Ayano\TDT_photometry\Tanks\k3546_k00_k00_k00-260107-154030'
    'Y:\Ayano\TDT_photometry\Tanks\k3549_k00_k00_k00-260107-162843'
};
reload = 0;
chunky_or_not = 0;

for t = 1:length(tank_dirs)
    % Extract mouse ID from tank folder name (e.g. "k3546_k00_k00_k00-..." -> "3546")
    [~, tank_folder] = fileparts(tank_dirs{t});
    parts = strsplit(tank_folder, '_');
    mouse_id = regexprep(parts{1}, '^k', '');
    Stem_Dir = fullfile('Z:\Koji\Neuropixels', mouse_id);

    sessions = dir(fullfile(tank_dirs{t}, 'Mouse-*'));
    sessions = sessions([sessions.isdir]);

    for s = 1:length(sessions)
        session_name = fullfile(tank_dirs{t}, sessions(s).name);

        % Extract date from session folder name (e.g. "Mouse-260107-155658" -> "260107")
        name_parts = strsplit(sessions(s).name, '-');
        date_str = name_parts{2};

        output_dir = fullfile(Stem_Dir, 'TDT', date_str);
        if ~exist(output_dir, 'dir'), mkdir(output_dir); end

        Save_univ_dir0 = output_dir;
        Save_univ_dir1 = output_dir;
        Save_univ_dir2 = output_dir;

        fprintf('\n===== [%d/%d] %s -> %s =====\n', ...
            s, length(sessions), sessions(s).name, output_dir);

        try
            TDT_demod(session_name, Save_univ_dir0, Save_univ_dir1, reload);
            TDT_dFF_stage2(session_name, Stem_Dir, Save_univ_dir0, Save_univ_dir1, Save_univ_dir2, chunky_or_not);
        catch ME
            fprintf('ERROR on %s: %s\n', sessions(s).name, ME.message);
        end
    end
end
