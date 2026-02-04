   function D_out = lowpass_butter_sos_gain1(D,Fs,lowpass_order,lowpass_cutoff__Hz)
        D_out = [];
        Wn = lowpass_cutoff__Hz/Fs*2;
        [z, p, k]      = butter(lowpass_order,Wn);
        sos = zp2sos(z, p, k);
        filtfunc = @(x) filtfilt(sos, 1, x);
        for ii = 1:size(D,2)
          D_out(:,ii) = filtfunc(D(:,ii));
        end
    end