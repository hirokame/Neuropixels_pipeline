function BombCell(ephysKilosortPath, ephysRawFile, savePath)
    % This function runs BombCell quality metrics on a single session.
    % It is called from the Python pipeline.
    %
    % ARGS:
    %   ephysKilosortPath (char): Path to the Kilosort output directory.
    %   ephysRawFile (char): Path to the raw ephys .bin file.
    %   savePath (char, optional): Path to save the output files. Defaults to ephysKilosortPath.

    if nargin < 3 || isempty(savePath)
        savePath = ephysKilosortPath;
    end

    [ephysRawFileDir, ephysRawFileName, ~] = fileparts(ephysRawFile);
    metaFile = fullfile(ephysRawFileDir, [ephysRawFileName, '.meta']);
    
    if ~exist(metaFile, 'file')
        error('BombCell:NoMetaFile', 'Could not find .meta file. Looked for: %s', metaFile);
    end
    ephysMetaDirStruct = dir(metaFile);
     
    % Ensure savePath exists
    if ~exist(savePath, 'dir')
        mkdir(savePath);
    end

    kilosortVersion = 4;
    gain_to_uV = NaN;

    [spikeTimes_samples, spikeClusters, templateWaveforms, templateAmplitudes, pcFeatures, ...
        pcFeatureIdx, channelPositions] = bc.load.loadEphysData(ephysKilosortPath, savePath);

    param = bc.qm.qualityParamValues(ephysMetaDirStruct, ephysRawFile, ephysKilosortPath, gain_to_uV, kilosortVersion);
    
    % Force internal saving paths to use the provided savePath
    param.ephysKilosortPath = savePath; 
    
    param.nChannels = 385;
    param.nSyncChannels = 1;
    param.maxPercSpikesMissing = 70; 
    param.minPresenceRatio = 0.500; 
    param.minNumSpikes = 200;
    param.hillOrLlobetMethod = 0;
    param.maxRPVviolations = 0.2000;

    if exist(metaFile, 'file')
        if endsWith(ephysMetaDirStruct(1).name, '.ap.meta') % SpikeGLX file-naming convention
            meta = bc.dependencies.SGLX_readMeta.ReadMeta(ephysMetaDirStruct(1).name, ephysMetaDirStruct(1).folder);
            [AP, ~, SY] = bc.dependencies.SGLX_readMeta.ChannelCountsIM(meta);
            param.nChannels = AP; % AP channels
            param.nSyncChannels = SY;
        end
    end

    [qMetric, unitType] = bc.qm.runAllQualityMetrics(param, spikeTimes_samples, spikeClusters, ...
    templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx, channelPositions, savePath);

    % (for use with another language: output a .tsv file of labels. You can then simply load this) 
    label_table = table(unitType);
    writetable(label_table, fullfile(savePath, 'templates._bc_unit_labels.tsv'), 'FileType', 'text', 'Delimiter', '\t');
    
    disp('BombCell analysis complete.');
end
