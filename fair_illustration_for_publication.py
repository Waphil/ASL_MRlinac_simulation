
# Add Functions Folder to Path
addpath(genpath('./functions'))

# Add Images to Matlab Path
addpath(genpath('C:\Data\Philip\Arbeit\USZ\MRI Data'))


## FAIR Illustration
##################################################################

# Read Folder
pathImg = ['C:\Data\Philip\Arbeit\USZ\MRI Data\MRIdian Scans\' ...
           'Session15_March21st\GRE_LOCALIZER_NCH_FOV500_0001'];
fileImg = dir(fullfile(pathImg,'**\*.IMA'));
fileImg = fileImg(1);

# Get Info
infoImg = dicominfo(fullfile(fileImg.folder,fileImg.name));

# Load & Crop Image
image   = dicomread(infoImg.Filename);
cropped = image(60:end-40,40:end-60);

# Create masks
imslice          = 50:54;
invsel           = 44:60;
invglob          = 15:140;
mask1            = zeros(size(cropped));
mask1(invsel,:)  = 1;
mask1(imslice,:) = 2;
mask2            = zeros(size(cropped));
mask2(invglob,:) = 1;
mask2(imslice,:) = 2;

# Plot
Background = ind2rgb(uint8(100*mat2gray(cropped)),flipud(gray(256)));
cmapinit   = flipud(hot(20));
cmap       = cmapinit([1,6,12],:);

figure
  tiledlayout(1,2,TileSpacing="tight");
  nexttile
  imshow(Background,[],InitialMagnification=600); hold on
  im2           = imshow(mask2,[0,2],colormap=cmap);
  im2.AlphaData = 0.4*isfinite(mask2);
  xlabel("Tag Image",FontSize=28,FontWeight="bold")
  nexttile
  imshow(Background,[],InitialMagnification=600); hold on
  im2           = imshow(mask1,[0,2],colormap=cmap);
  im2.AlphaData = 0.4*isfinite(mask1);
  xlabel("Control Image",FontSize=28,FontWeight="bold")
  x1  = scatter(nan, nan,'s',MarkerFaceColor=cmapinit(6,:), ...
                MarkerEdgeColor=cmapinit(6,:),MarkerFaceAlpha=0.4, ...
                MarkerEdgeAlpha=0.4,LineWidth=15);
  x2  = scatter(nan, nan,'s',MarkerFaceColor=cmapinit(13,:), ...
                MarkerEdgeColor=cmapinit(13,:),MarkerFaceAlpha=0.4, ...
                MarkerEdgeAlpha=0.4,LineWidth=15);
  lgd = legend([x1,x2],["Inversion slice","Imaging slice"], ...
               FontSize=24,Orientation='horizontal',Box='off');
  lgd.Layout.Tile = 'north';
