function [x_p] = processHOGrad(x, window_size, center, p_norm)
%Process Hogs that are located on the center vector
%They are stored with the order given by the center's argument (origin :
%center of the image)
    x = reshape(x,[size(x,1),32,32]);
    l = size(center,1);
    d = idivide(uint8(window_size),2);
    n_bins = 12;
    x_p = zeros(size(x,1),n_bins*l);
    edges = (1:n_bins+1);
    %Sorting centers by angle
    center_translated = center - repmat([16.5 16.5],size(center,1),1);
    angle_centers = angle(center_translated(:,1)+1i*center_translated(:,2));
    [sorted,sorted_ind] = sort(angle_centers);
    center = center(sorted_ind,:);
    for i=1:size(x,1)
        image = squeeze(x(i,:,:));
        [~,Gdir] = imgradient(image);
        Gdir(Gdir<0) = Gdir(Gdir<0)+360;
        Gdir_bin = ceil(n_bins*Gdir/360);
        for c=1:length(center)
            window_origin = [max(1,min(32-window_size,center(c,1)-d)), max(1,min(32-window_size,center(c,2)-d))];
            window = Gdir_bin(window_origin(1):window_origin(1)+window_size-1,window_origin(2):window_origin(2)+window_size-1);
            histogram = histcounts(window,edges);
            x_p(i,(c-1)*n_bins+1:c*n_bins) = histogram;
        end
        % normalization
        if p_norm~=0
            x_p(i,:) = x_p(i,:)/norm(x_p(i,:), p_norm);
        end
    end
end
