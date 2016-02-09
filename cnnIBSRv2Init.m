function net = cnnIBSRv2Init(varargin)
% CNN_IMAGENET_INIT  Baseline CNN model

opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.nLabels = 39;
opts = vl_argparse(opts, varargin) ;

net.layers = {} ;

% Define input and output size
net.normalization.imageSize = [256, 256, 1] ;
net.normalization.interpolation = 'bilinear' ;
net.normalization.averageImage = [] ;
net.normalization.keepAspect = true ;
net.backPropDepth = inf;
net.outputSize = [64 64]; % height and width of the final convolutional layer
net.nLabels = opts.nLabels;

% Block 1
net = addConvBlock(net, opts, 1, 7, 7, net.normalization.imageSize(3), 64, 1, 3, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 1) ;

% Block 2
net = addConvBlock(net, opts, 2, 5, 5, 64, 128, 1, 2, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 1) ;

% Block 3
net = addConvBlock(net, opts, 3, 3, 3, 128, 256, 1, 2, 2) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool3', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 1, ...
                           'pad', 1) ;
net.layers{end+1} = struct('type', 'dropout', 'name', 'dropout3', 'rate', 0.5) ;

% Block 4
net = addConvBlock(net, opts, 4, 3, 3, 256, 512, 1, 2, 2) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool4', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 1, ...
                           'pad', 1) ;
net.layers{end+1} = struct('type', 'dropout', 'name', 'dropout4', 'rate', 0.5) ;

% Block 5
net = addConvBlock(net, opts, 5, 3, 3, 512, 512, 1, 2, 2) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 1, ...
                           'pad', 1) ;
net.layers{end+1} = struct('type', 'dropout', 'name', 'dropout5', 'rate', 0.5) ;

% Block 6
net = addConvBlock(net, opts, 6, 4, 4, 512, 1024, 1, 6, 4) ;
net.layers{end+1} = struct('type', 'dropout', 'name', 'dropout6', 'rate', 0.5) ;


% Block 7
net = addConvBlock(net, opts, 7, 1, 1, 1024, opts.nLabels, 1, 0, 1) ;
net.layers(end) = [] ;  % remove last relu layer

% Block 8
net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;


function net = addConvBlock(net, opts, id, h, w, in, out, stride, pad, hole)
if nargin < 10, hole = 1; end
if nargin < 9, pad = 0; end
if nargin < 8, stride = 1; end
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
  name = 'fc' ;
else
  name = 'conv' ;
end
if ischar(id)
    convName = sprintf('%s%s', name, id);
    reluName = sprintf('relu%s',id);
else
    convName = sprintf('%s%d', name, id);
    reluName = sprintf('relu%d',id);
end
net.layers{end+1} = struct('type', 'conv', 'name', convName, ...
                           'weights', {{0.01/opts.scale * randn(h, w, in, out, 'single'), zeros(1, out,'single')}}, ...
                           'stride', stride, ...
                           'pad', pad, ...
                           'hole', hole,...
                           'learningRate', [1 2], ...
                           'weightDecay', [opts.weightDecay 0]) ;
net.layers{end+1} = struct('type', 'relu', 'name', reluName) ;
