% function [daily_return, total_return] = CAEGc_run(data)
%{
This file is the run core for the Aggregating exponential gradient expert 
advice for online portfolio selection under transaction costs (CAEGc).

For any usage of this function, the following papers should be cited as
reference:

[1] Yong Zhang, Jiahao Li, Xingyu Yang, and Hong Lin. "Aggregating 
exponential gradient expert advice for online portfolio selection under 
transaction costs" Journal of the Operational Research Society, 2022.
[2] Xingyu Yang, Jin'an He, and Yong Zhang. "Aggregating exponential 
gradient expert advice for online portfolio selection" Journal of the 
Operational Research Society, 2022, 73(3): 587-597.

At the same time, it is encouraged to cite the following papers with 
previous related works:

[3] Jiahao Li, Yong Zhang, Xingyu Yang, and Liangwei Chen. "Online 
portfolio management via deep reinforcement learning with high-frequency 
data" Information Processing & Management, 2023, 60(3): 103247.

Inputs:
data                      -data with price relative sequences

Outputs:
daily_return              -daily wealths
total_return              -total wealths
%}

%% Parameter Setting
tc = 0; % transaction cost rate

%% Variables Inital
[T,N] = size(data);
b = zeros(T,N);
daily_return = zeros(T,1);
total_return = zeros(T,1);
eta_min = 0.01;
step = 0.01;
eta_max = 0.2;

s = cell(1,N);
for ns = 1:N
    s{1,ns} = 0;
end

num_eta=0;
for eta = eta_min:step:eta_max
    num_eta = num_eta+1;
end

S = cell(1,N);
for nS = 1:N
    S{1,nS} = ones(num_eta,1);
end

S_o = cell(1,N);
for nS = 1:N
    S_o{1,nS} = ones(num_eta,1);
end

e = cell(T,num_eta);
h = ones(1,N)/N;
for i = 1:num_eta
    e{1,i} = h;
end

%% Main
for t = 1:T
    if t==1
        b(1,:) = ones(1,N)/N;
        daily_return(t,1) = b(t,:)*data(t,:)';
        total_return(t) = b(t,:)*data(t,:)';

%         daliy_exp_r(:,:,t) = b(:,:)*data(t,:)';
%         exp_cumres(:,:,t) = daliy_exp_r(:,:,t);

    else 
        k=0;
        for eta = eta_min:step:eta_max
            k = k+1;
            ff = e{t-1,k};
            Z = ff.*exp(eta*data(t-1,:)/(ff*data(t-1,:)'))*ones(N,1);
            f = ff.*exp(eta*data(t-1,:)/(ff*data(t-1,:)'));
            f = f/Z;
            e{t,k} = f;

            exp_h(t-1,:)=data(t-1,:).*e{t-1,k};
            exp_hat(t-1,:)=exp_h(t-1,:)/sum(exp_h(t-1,:));
            exp_diff(t,:)=sum(abs(e{t,k}-exp_hat(t-1,:)));

            for n1 = 1:N
                if t <= 2
                    S{1,n1}(k,1) = S{1,n1}(k,1)*(ff*data(t-1,:)');
                else
                    c_min=(tc*exp_diff(t-1,:))/(1+tc);
                    c_max=(tc*exp_diff(t-1,:))/(1-tc);
                    [c,~]=fminbnd(@(c) abs(S{1,n1}(k,1)*c-tc*S{1,n1}(k,1)*...
                        sum(abs(ff*(1-c)-exp_hat(t-2,:)))),c_min,c_max);
                    S{1,n1}(k,1) = S{1,n1}(k,1)*(ff*data(t-1,:)'*(1-c));
                end
            end

            s{1,N} = s{1,N}+S{1,N}(k,1)^(1/sqrt(t));
            for n2 = 1:N-1
                s{1,n2} = s{1,n2}+S{1,n2}(k,1)^(1/sqrt(t))*f(:,n2);
            end
        end

        SUM_N = 0;
        for n3 = 1:N
            if n3 < N
                b(t,n3) = s{1,n3}/s{1,N};
                SUM_N = SUM_N+s{1,n3};
            else
                b(t,n3) = (s{1,N}-SUM_N)/s{1,N};
            end
        end

        b_h(t-1,:) = data(t-1,:).*b(t-1,:);
        b_hat(t-1,:) = b_h(t-1,:)/sum(b_h(t-1,:));
        diff(t,:) = sum(abs(b(t,:)-b_hat(t-1,:)));
        
        c_min(t,:) = (tc*diff(t,:))/(1+tc);
        c_max(t,:) = (tc*diff(t,:))/(1-tc);
        [c,~] = fminbnd(@(c) abs(total_return(t-1,:)*c-tc*...
            total_return(t-1,:)*sum(abs(b(t,:)*(1-c)-b_hat(t-1,:)))),...
            c_min(t,:),c_max(t,:));
        c1(t,1) = c;
        
        daily_return(t) = data(t,:)*b(t,:)'*(1-c);
        total_return(t) = total_return(t-1)*daily_return(t);
    end
end
