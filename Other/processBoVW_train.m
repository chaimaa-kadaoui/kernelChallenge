function [x_p, C, visual_words] = processBoVW_train(x, window_size, stride, p_norm, n_clusters, max_iter, display_silhouette)
    if nargin < 7
        display_silhouette = 0;
    end
%     x = reshape(x,[size(x,1),32,32]);
%     l = floor((32-window_size)/stride)+1;
%     visual_words = zeros(size(x,1)*l^2,window_size^2);
%     for i=1:size(x,1)
%         image = squeeze(x(i,:,:));
%         for row=1:l
%             for col=1:l
%                 window = image((row-1)*stride+(1:window_size),(col-1)*stride+(1:window_size));
%                 visual_words((i-1)*l^2+(row-1)*l+col,:) = window(:);
%             end
%         end
%     end

%     x = reshape(x,[size(x,1),32,32]);
%     l = floor((32-window_size)/stride)+1;
%     n_bins = 12;
%     x_p = zeros(size(x,1),l,l,n_bins);
%     edges = (1:n_bins+1);
%     for i=1:size(x,1)
%         image = squeeze(x(i,:,:));
%         [~,Gdir] = imgradient(image);
%         Gdir(Gdir<0) = Gdir(Gdir<0)+360;
%         Gdir_bin = ceil(n_bins*Gdir/360);
%         parfor row=1:l
%             for col=1:l
%                 window = Gdir_bin((row-1)*stride+(1:window_size),(col-1)*stride+(1:window_size));
%                 histogram = histcounts(window,edges);
%                 %index = size(x,1)*l*i+l*row+col;
%                 x_p(i,row,col,:) = histogram;
%                 %x_p(i,(row-1)*l*n_bins+(n_bins*(col-1)+1:n_bins*col)) = histogram;
%             end
%         end
%         % normalization
%         if p_norm~=0
%             x_p(i,:) = x_p(i,:)/norm(x_p(i,:), p_norm);
%         end
%         fprintf('Progression : %f%%\n',i/size(x,1));
%     end

    x_p = load('histograms_w_8_s_2.mat');
    visual_words = x_p.x_new;

    visual_words_train = x_p.x_new(1:4:end,:);
    fprintf('Computing k-means clustering from %i visual words\n',size(visual_words,1));
    [idx_train, C] = kmeans(visual_words_train, n_clusters,'Display','iter','MaxIter',max_iter);
    %C = load('C_1000.mat');
    %C = C.C;
    %D = dist([visual_words;C]);
    %[distance, idx] = min(D(1,2:end));
    
    if display_silhouette
        figure;
        silhouette(visual_words,idx);
        xlabel 'Silhouette Value'
        ylabel 'Cluster'
    end
    
    x = reshape(x,[size(x,1),32,32]);
    l = floor((32-window_size)/stride)+1;
    n_bins = 12;
    x_p = reshape(x_p.x_new,size(x,1),l,l,n_bins);
    
    %For each quarter : [begin_x,end_x,begin_y,end_y,feature_offset]
    quarter_param = [[1,floor(l/2),1,floor(l/2),0];
        [floor(l/2)+1,l,1,floor(l/2),n_clusters];
        [1,floor(l/2),floor(l/2)+1,l,2*n_clusters];
        [floor(l/2)+1,l,floor(l/2)+1,l,3*n_clusters]];
    
    x_classes = zeros(size(x,1),4*n_clusters);
    edges = (1:n_clusters+1);
    for i=1:size(x,1)
        %For each quarter, we process the histogram
        for q = 1:size(quarter_param,1)
            curr_histo = zeros(1,n_clusters);
            for j=quarter_param(q,1):quarter_param(q,2)
                for k = quarter_param(q,3):quarter_param(q,4)
                    [min_dist,cur_class] = min(sum((squeeze(repmat(x_p(1,1,1,:),n_clusters,1))-C).^2,2));
                    curr_histo(1,cur_class) = curr_histo(cur_class) + 1;
                end
            end
            % normalization
            if p_norm~=0
                x_classes(i,floor(quarter_param(q,5)+1):floor(quarter_param(q,5)+n_clusters)) = curr_histo(1,:)/norm(curr_histo(1,:), p_norm);
            end
        end
        if mod(i,100) == 0
            fprintf('Progression : %f%%\n',i/size(x,1));
        end
    end
end