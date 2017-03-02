function [x_p, C, visual_words] = processBoVW_train(x, window_size, stride, p_norm, n_clusters, max_iter, display_silhouette)
    if nargin < 7
        display_silhouette = 0;
    end
    x = reshape(x,[size(x,1),32,32]);
    l = floor((32-window_size)/stride)+1;
    visual_words = zeros(size(x,1)*l^2,window_size^2);
    for i=1:size(x,1)
        image = squeeze(x(i,:,:));
        for row=1:l
            for col=1:l
                window = image((row-1)*stride+1:row*stride,(col-1)*stride+1:col*stride);
                visual_words((i-1)*l^2+(row-1)*l+col,:) = window(:);
            end
        end
    end
    fprintf('Computing k-means clustering from %i visual words\n',size(visual_words,1));
    [idx, C] = kmeans(visual_words, n_clusters,'Display','iter','MaxIter',max_iter);
    
    if display_silhouette
        figure;
        silhouette(visual_words,idx);
        xlabel 'Silhouette Value'
        ylabel 'Cluster'
    end
    
    x_p = zeros(size(x,1),n_clusters);
    edges = (1:n_clusters+1);
    for i=1:size(x,1)
        x_p(i,:) = histcounts(idx((i-1)*l^2+1:i*l^2), edges);
        % normalization
        if p_norm~=0
            x_p(i,:) = x_p(i,:)/norm(x_p(i,:), p_norm);
        end
    end
end