%%
%% Q3
% subject1
clc; clear; close all;

load 'subject1.mat'

new_Subject1 = transpose(table2array(subject1));
new_Subject1 = new_Subject1(1:19,:);

save new_Subject1.mat new_Subject1

EEG.etc.eeglabvers = '2023.0'; % this tracks which version of EEGLAB is being used, you may ignore it
     EEG = pop_importdata('dataformat','matlab','nbchan',0,'data','C:\\Users\\gilan\\OneDrive\\Documents\\MATLAB\\eeglab2023.0\\new_Subject1.mat','setname','subject1','srate',200,'pnts',0,'xmin',0);
     EEG = pop_reref( EEG, []);
     EEG.setname='subject1_reref';
     EEG = pop_eegfiltnew(EEG, 'locutoff',0.5,'hicutoff',40.5,'plotfreqz',1);
     EEG.setname='subject1_reref_filter';
     EEG = pop_editset(EEG, 'chanlocs', 'C:\\Users\\gilan\\OneDrive\\Documents\\MATLAB\\eeglab2023.0\\sample_locs\\Standard-10-20-Cap19.locs');
     EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'interrupt','on');
     EEG = pop_iclabel(EEG, 'default');
     EEG = pop_subcomp( EEG, [1   2   3   4   5   7  19], 0);
     EEG = eeg_checkset( EEG );
     EEG = pop_iclabel(EEG, 'default');


Data = EEG.data; % Loading the Data from EEG Workspace
Start = 14*200; % Starting sample - from 14 sec after start of the event
End = 1214*200; % Ending Sample - Neglecting the data after 120 trials
Clean_Data = Data(:, Start:End);

Epoch = zeros(19,600,120);

for i = 1:19
    for j = 1:120
        Epoch( i , : , j) = Clean_Data(i, (j-1)*2000 + 601 : (j-1)*2000 + 1200);
    end
end

save epoch1.mat Epoch

EEG.etc.eeglabvers = '2023.0'; % this tracks which version of EEGLAB is being used, you may ignore it
     EEG = pop_importdata('dataformat','matlab','nbchan',0,'data','C:\\Users\\gilan\\OneDrive\\Documents\\MATLAB\\eeglab2023.0\\epoch1.mat','setname','epoch1','srate',1,'pnts',0,'xmin',0);
     pop_eegplot( EEG, 1, 1, 1);

Cleaned_Epoch = EEG.data;
Subsampled_Epoch = Cleaned_Epoch([1, 5, 10, 15], : , : ); % The Final Epoch

Epoch_Struct = struct('Epoch_Subsampled', Subsampled_Epoch);

save epoch1_struct Epoch_Struct

%% subject2
clc; clear; close all;

load 'subject2.mat'

new_Subject2 = transpose(table2array(subject2));
new_Subject2 = new_Subject2(1:19,:);

save new_Subject2.mat new_Subject2

EEG.etc.eeglabvers = '2023.0'; % this tracks which version of EEGLAB is being used, you may ignore it
     EEG = pop_importdata('dataformat','matlab','nbchan',0,'data','C:\\Users\\gilan\\OneDrive\\Documents\\MATLAB\\eeglab2023.0\\new_Subject2.mat','setname','subject2','srate',200,'pnts',0,'xmin',0);
     EEG = pop_reref( EEG, []);
     EEG.setname='subject2_reref';
     EEG = pop_eegfiltnew(EEG, 'locutoff',0.5,'hicutoff',40.5,'plotfreqz',1);
     EEG.setname='subject2_reref_bandpass';
     EEG = pop_editset(EEG, 'chanlocs', 'C:\\Users\\gilan\\OneDrive\\Documents\\MATLAB\\eeglab2023.0\\sample_locs\\Standard-10-20-Cap19.locs');
     EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'interrupt','on');
     EEG = pop_iclabel(EEG, 'default');
     EEG = pop_subcomp( EEG, [5   9  19], 0);
     EEG = eeg_checkset( EEG );
     EEG = pop_iclabel(EEG, 'default');


Data = EEG.data; % Loading the Data from EEG Workspace
Start = 14*200; % Starting sample - from 14 sec after start of the event
End = 1214*200; % Ending Sample - Neglecting the data after 120 trials
Clean_Data = Data(:, Start:End);

Epoch = zeros(19,600,120);

for i = 1:19
    for j = 1:120
        Epoch( i , : , j) = Clean_Data(i, (j-1)*2000 + 601 : (j-1)*2000 + 1200);
    end
end

save epoch2.mat Epoch
%% just for show
EEG.etc.eeglabvers = '2023.0'; % this tracks which version of EEGLAB is being used, you may ignore it
     EEG = pop_importdata('dataformat','matlab','nbchan',0,'data','C:\\Users\\gilan\\OneDrive\\Documents\\MATLAB\\eeglab2023.0\\epoch2.mat','setname','epoch2','srate',1,'pnts',0,'xmin',0);
     pop_eegplot( EEG, 1, 1, 1);
     EEG.setname='epoch2_filtered';
     EEG = eeg_checkset( EEG );

%% subject2
Cleaned_Epoch = EEG.data;
Subsampled_Epoch = Cleaned_Epoch([1, 5, 10, 15], : , : ); % The Final Epoch

Epoch_Struct = struct('Epoch_Subsampled', Subsampled_Epoch);

save epoch2_struct Epoch_Struct

%% Q4
%% Q4-1
clc; clear; close all;

load 'AD.mat' 
load 'normal.mat'

Normal_Data = zeros(15,2); % First column for Frequent and Second for Rare
AD_Data = zeros(13,2); % First column for Frequent and Second for Rare

% normal
for n = 1:1:15
    Epoch = normal(n).epoch;
    Odor = normal(n).odor;
    Normal_Data(n, : ) = PLV_Calculator(Epoch, Odor);
end

% AD
for n = 1:1:13
    Epoch = AD(n).epoch;
    Odor = AD(n).odor;
    AD_Data(n, : ) = PLV_Calculator(Epoch, Odor);
end

subplot(2,2,1)
boxplot(Normal_Data(:,1));
title('PLV for Normal patients and Frequent Odor');
grid minor
subplot(2,2,2)
boxplot(Normal_Data(:,2));
title('PLV for Normal patients and Rare Odor');
grid minor
subplot(2,2,3)
boxplot(AD_Data(:,1));
title('PLV for AD patients and Frequent Odor');
grid minor
subplot(2,2,4)
boxplot(AD_Data(:,2));
title('PLV for AD patients and Rare Odor');
grid minor

%% Q4-2
Fit_Dist1 = fitdist(Normal_Data(:,1),"Normal");
Fit_Dist2 = fitdist(Normal_Data(:,2),"Normal");
Fit_Dist3 = fitdist(AD_Data(:,1),"Normal");
Fit_Dist4 = fitdist(AD_Data(:,2),"Normal");
x = -0.5:0.01:2;
pdf1 = pdf(Fit_Dist1,x);
pdf2 = pdf(Fit_Dist2,x);
pdf3 = pdf(Fit_Dist3,x);
pdf4 = pdf(Fit_Dist4,x);

% Normal fits
subplot(2,2,1)
plot(x,pdf1);
title('Normal fit to PLV Normal patient and Frequent Odor');
grid minor
subplot(2,2,2)
plot(x,pdf2);
title('Normal fit to PLV Normal patient and Rare Odor');
grid minor
subplot(2,2,3)
plot(x,pdf3);
title('Normal fit to PLV AD patient and Frequent Odor');
grid minor
subplot(2,2,4)
plot(x,pdf4);
title('Normal fit to PLV AD patient and Rare Odor');
grid minor

% P-Values
[h1,p1] = ttest2(Normal_Data(:,1),AD_Data(:,1));
[h2,p2] = ttest2(Normal_Data(:,2),AD_Data(:,2));


%% functions

function PLV = PLV_Finder(data, channel1, channel2)

    % Extract the time series data for the two channels
    signal1 = data(channel1, :);
    signal2 = data(channel2, :);

    % Compute the Fourier transform of the signals
    fftSignal1 = fft(signal1);
    fftSignal2 = fft(signal2);

    % Define the frequency axis for FFT
    freqAxis = (0:length(signal1)-1)*(200/length(signal1));

    % Find indices corresponding to the desired frequency range
    freqIndices = find(freqAxis >= 35 & freqAxis <= 40);

    % Extract the phase angles for each frequency bin within the range
    phaseSignal1 = angle(fftSignal1(freqIndices));
    phaseSignal2 = angle(fftSignal2(freqIndices));

    % Compute PLV as mean of phase differences across frequencies
    PLV = abs(mean(exp(1i * (phaseSignal1 - phaseSignal2))));
end

function PLVPP = PLV_Calculator(Epoch, Odor)
    Num_of_Frequent = size(Odor,1) - sum(Odor);
    
    % Loop to find the PLV for each trial

    % Defining an array to store the PLV values for each patient
    PLV_Array = zeros(1,size(Epoch,3));

    for i = 1 : size(Epoch,3)
        PLV_Array(1,i) = PLV_Finder(Epoch(:,:,i),2,3);
    end

    % Defining vars to store the sum of different exposures
    Sum_Frequent = 0;
    Sum_Rare = 0;

    % Dividing Frequent and Rare exposures of each trial
    for i = 1:size(Odor,1)
        if Odor(i,1) == 0
            Sum_Frequent = Sum_Frequent + PLV_Array(1,i);
        end
        if Odor(i,1) == 1
            Sum_Rare = Sum_Rare + PLV_Array(1,i);
        end
    end

    % Finding the Average PLV for Frequent and Rare exposures
    PLV_Frequent = Sum_Frequent / Num_of_Frequent;
    PLV_Rare = Sum_Rare / (size(Odor,1) - Num_of_Frequent);
    PLVPP = [PLV_Frequent, PLV_Rare];
end

