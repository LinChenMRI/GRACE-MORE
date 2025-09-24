segLen = 300;
num_segments = floor(length(resp) / segLen);
freqs = zeros(1, length(resp));

for k = 1:num_segments
    idx = (k - 1) * segLen + (1:segLen);
    Res_Signal = resp(idx);
    Res_Signal = Res_Signal - min(Res_Signal);
    Res_Signal = Res_Signal ./ max(Res_Signal);
    Res_Signal = sgolayfilt(Res_Signal, 1, 3);

    IMFs = emd(Res_Signal);
    mmin = min(IMFs);
    mmax = max(IMFs);
    normalized_IMFs = (IMFs - mmin) ./ (mmax - mmin);

    IMFs_Freq = normalized_IMFs(:, 1);
    [cwt_result, ~] = cwt(IMFs_Freq, 'bump');
    main_freqs_list = sum(abs(cwt_result).^2).';

    a = min(main_freqs_list);
    b = max(main_freqs_list);
    freqs(idx) = (main_freqs_list - a) ./ (b - a);
end
