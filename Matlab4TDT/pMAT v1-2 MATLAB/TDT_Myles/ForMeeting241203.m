F1 = handles.data.streams.Fi1r.data(1,:); % sensor 1
F2 = handles.data.streams.Fi1r.data(2,:); % sensor 2
Fs = handles.data.streams.Fi1r.fs;

frqs_405 = 210;
frqs_465 = 330;
frqs_560 = 530;

[ydemod405_1, cfg, figs] = quickdemod(F1,Fs,frqs_405);
[ydemod405_2, cfg, figs] = quickdemod(F2,Fs,frqs_405);
[ydemod465_1, cfg, figs] = quickdemod(F1,Fs,frqs_465);
[ydemod465_2, cfg, figs] = quickdemod(F2,Fs,frqs_465);
[ydemod560_1, cfg, figs] = quickdemod(F1,Fs,frqs_560);
[ydemod560_2, cfg, figs] = quickdemod(F2,Fs,frqs_560);

onlinedemod405 = handles.Ch405A;
onlinedemod465 = handles.Ch465A;
onlinedemod560 = handles.Ch560B;

for pair_n = 1:6
    switch pair_n
        case 1
            Signal  = ydemod465_1;
            Control = ydemod405_1;
        case 2
            Signal  = ydemod560_2;
            Control = ydemod405_1;
        case 3
            Signal  = ydemod560_2;
            Control = ydemod405_2;
        case 4
            Signal  = ydemod560_2;
            Control = ydemod465_2;
        case 5
            Signal  = onlinedemod465;
            Control = onlinedemod405;
        case 6
            Signal  = onlinedemod560;
            Control = onlinedemod405;
    end
    dFF_raw = DeltaF(Signal,Control,Smoothing_Value);
end