function [x_p] = processHOG(x, window_size, stride)
    x = reshape(x,[size(x,1),32,32]);
    l = floor((32-window_size)/stride)+1;
    n_bins = 12;
    x_p = zeros(size(x,1),n_bins*l^2);
    edges = (1:n_bins+1);
    for i=1:size(x,1)
        image = squeeze(x(i,:,:));
        [~,Gdir] = imgradient(image);
        Gdir(Gdir<0) = Gdir(Gdir<0)+360;
        Gdir_bin = ceil(n_bins*Gdir/360);
        for row=1:l
            for col=1:l
                window = Gdir_bin((row-1)*stride+1:row*stride,(col-1)*stride+1:col*stride);
                histogram = histcounts(window,edges);
                x_p(i,(row-1)*l*n_bins+(n_bins*(col-1)+1:n_bins*col)) = histogram;
            end
        end
        % L2 normalization
        x_p(i,:) = x_p(i,:)/norm(x_p(i,:));
    end
end