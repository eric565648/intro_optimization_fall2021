% initialization
clear all; close all;

x=[-4 -3 -2 -1 0 1 2 3 4 5 6];
y=[56 35 21 11 3 1 0.5 6 13 28 48]';
% A matrix
A=[];
for i=x
    A=[A; i^2 i 1];
end
Q=A'*A;

max_k=50;

for j=1:6

% randomize a initial state between -10~10
x_0=[rand*20-10;rand*20-10;rand*20-10];

x_g=[x_0]; %state of gradient descent
x_s=[x_0]; %state of steepest descent
x_b=[x_0]; %state of back tracking

error_g=[];
error_s=[];
error_b=[];

% gradient descent parameters
a_g = 1/max(svd(A'*A));

% Steepest descent parameters

% Backtracking parameters
c = 0.25; beta=0.5; a0=1;

for i=1:max_k
%     if i==1
%         xgk=x_0;
%         xsk=x_0;
%         xbk=x_0;
%     else
%         xgk=x_g(:,end);
%         xsk=x_s(:,end);
%         xbk=x_b(:,end);
%     end
    xgk=x_g(:,end);
    xsk=x_s(:,end);
    xbk=x_b(:,end);

    % gradient descent
    xgk1 = xgk-a_g*dh(xgk,A,y);
    x_g = [x_g xgk1];
    % Steepest Descent
    gemma=dh(xsk,A,y);
    ask=(gemma'*gemma)/(gemma'*A'*A*gemma);
    xsk1 = xsk-ask*gemma;
    x_s = [x_s xsk1];
    % Backtracking
    pk=-dh(xbk,A,y); ab=a0;
    while h(xbk+ab*pk,A,y)>h(xbk,A,y)-c*ab*norm(pk)^2
        ab=ab*beta;
    end
    xbk1=xbk+ab*pk;
    x_b = [x_b xbk1];
    
    error_g = [error_g h(xgk1,A,y)];
    error_s = [error_s h(xsk1,A,y)];
    error_b = [error_b h(xbk1,A,y)];
end

% figure(1)
% plot3([x_g(1,:); x_s(1,:); x_b(1,:)]',[x_g(2,:); x_s(2,:); x_b(2,:)]',[x_g(3,:); x_s(3,:); x_b(3,:)]');
% title('Gradient descent based method comparison');
% legend('Gradient', 'Steepest', 'Backtrack'); grid;
% ylabel(''); xlabel(''); 

figure(j)
plot([1:max_k; 1:max_k; 1:max_k]',[error_g; error_s; error_b]');
title('Function output (error)');
legend('Gradient', 'Steepest', 'Backtrack'); ylabel('output (error)'); xlabel('iteration'); grid;

end

function fx = h(x,A,y)
    fx = 0.5*(x'*A'*A*x-x'*A'*y-y'*A*x+y'*y);
end

function d=dh(x,A,y)
    d=A'*A*x-A'*y;
end