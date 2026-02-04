% Add seqNMF to the MATLAB path
addpath(genpath('C:\Users\kouhi\Downloads\seqNMF-master\seqNMF-master'));

% Set the kilosort4 data directory
ks_dir = 'Z:\Koji\9153_01292025_tagging_g0\9153_01292025_tagging_g0_imec0\kilosort4\';

% Load the binned spikes matrix
load([ks_dir 'binned_spikes_masked.mat']);  % This loads variable X

% Set seqNMF parameters, use existing values if provided
if ~exist('K', 'var'), K = 10; end
if ~exist('L', 'var'), L = 10; end
if ~exist('lambda', 'var'), lambda = 0; end

% Run seqNMF
[W, H, cost, loadings, power] = seqNMF(X, 'K', K, 'L', L, 'lambda', lambda, 'maxIter', 100);

% Optional: Visualize motifs
figure;
for k = 1:K
    subplot(K,1,k);
    imagesc(squeeze(W(:,:,k)));
    colorbar;
    ylabel('Neuron');
    xlabel('Time (bins)');
    title(['Motif ' num2str(k)]);
end

% Save results in the kilosort4 folder
save([ks_dir 'seqNMF_results.mat'], 'W', 'H', 'cost', 'loadings', 'power');