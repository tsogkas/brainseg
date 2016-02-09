function paths = setPaths()

paths.root         = fileparts(mfilename('fullpath'));
paths.results      = fullfile(paths.root,'results');
paths.models       = fullfile(paths.root,'models');
paths.data         = fullfile(paths.root, 'data');
paths.IBSR         = fullfile(paths.data, 'IBSR_nifti_stripped');
% paths.IBSR.images  = fullfile(paths.data, 'IBSR','orig');
% paths.IBSR.labels  = fullfile(paths.data, 'IBSR','seg');
