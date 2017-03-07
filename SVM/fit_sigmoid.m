% Adaptation of the pseudo-code found in the paper:
% Probabilistic Outputs for Support Vector Machines and
% Comparisons to Regularized Likelihood Methods
% by John C. Platt
% on March 26, 1999

function [A, B] = fit_sigmoid(score, y_bin)
    % score: SVM output Nx1 array
    % y_bin: true classes Nx1 with +1 and -1 values

    % Dimension
    N = size(y_bin,1);

    target = y_bin == 1;
    prior1 = sum(y_bin == 1);
    prior0 = sum(y_bin == -1);

    % Initialisation
    A = 0;
    B = log((prior0+1)/(prior1+1));
    hiTarget = (prior1+1)/(prior1+2);
    loTarget = 1/(prior0+2);
    lambda = 1e-3;
    olderr = 1e300;
    % pp is a temporary array
    pp = ((prior1+1)/(prior0+prior1+2))*ones(N,1);

    % Algorithm
    count = 0;
    for it = 1:100
        % We compute the Hessian & Gradient of error function w.r.t to A&B
        t = zeros(N,1);
        t(target) = hiTarget;
        t(~target) = loTarget;
        d1 = pp - t;
        d2 = pp.*(1-pp);
        a = sum((score.^2).*d2);
        b = sum(d2);
        c = sum(score.*d2);
        d = sum(score.*d1);
        e = sum(d1);
        % If the gradient is small we stop
        if (abs(d) < 1e-9 && abs(e) < 1e-9)
            break
        end
        oldA = A;
        oldB = B;
        err = 0;
        % We loop until the goodness of fit increases
        while true
            det = (a+lambda)*(b+lambda)-c^2;
            if (det==0)
                % If determinant of Hessian is 0 we increase the stabilizer
                lambda = lambda*10;
                continue
            end
            A = oldA + ((b+lambda)*d-c*e)/det;
            B = oldB + ((a+lambda)*e-c*d)/det;
            % The goodness of fit
            err = 0;
            p = 1./(1 + exp(A*score+B));
            pp = p;
            p(p==0) = exp(-200); % To make sure log(0) = -200
            p(p==1) = 1 - exp(-200);
            err = -sum(t.*log(p)+(1-t).*log(1-p));
            if err < olderr*(1 + 1e-7)
                lambda = lambda*0.1;
                break
            end
            % If error didn't increase: we increase the stabilizer*10
            lambda = lambda*10;
            if (lambda >= 1e6) % We give up...
                break
            end
        end
        diff = err-olderr;
        scale = 0.5*(err+olderr+1);
        if (diff > -1e-3*scale) && (diff < 1e-7*scale)
            count = count + 1;
        else
            count = 0;
        end
        olderr = err;
        if count == 3
            break
        end
    end

end
