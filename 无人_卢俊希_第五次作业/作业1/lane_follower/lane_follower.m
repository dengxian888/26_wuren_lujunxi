%% 第二问：轨迹跟踪
clear; clc; close all

% 车辆参数
lfr = 2.168 + 1.907; % 轴距 L
dt = 0.01;
v = 15; 
sim_steps = 2000;
Ld=10;%前视距离

% 参考轨迹 (正弦曲线)
X_ref = 0:0.1:200; 
Y_ref = 10 * sin(X_ref / 15); 

% 初始车辆状态 
X = X_ref(1); Y = Y_ref(1) + 3; phi = 0; 
X_vec = zeros(1, sim_steps); Y_vec = zeros(1, sim_steps);


for ii = 1:sim_steps
    X_vec(ii) = X; Y_vec(ii) = Y;
    
    
    % ===============================================================
    
    % ================= TODO 2.1: 实现纯跟踪算法 =================
    
    % 1. 查找前视目标点：在参考轨迹里，找到距离当前车刚好>=前视距离Ld的点
dist = sqrt((X_ref - X).^2 + (Y_ref - Y).^2); % 计算当前车到所有参考点的距离

[~, nearest_idx] = min(dist);  % 找最近点
ahead_idx = find(dist(nearest_idx:end) >= Ld, 1, 'first');

if isempty(ahead_idx)
    target_idx = length(X_ref);  % 找不到就用终点
else
    target_idx = ahead_idx + nearest_idx - 1;
end

X_target = X_ref(target_idx);                 % 目标点的全局X坐标
Y_target = Y_ref(target_idx);                 % 目标点的全局Y坐标

    % 2. 坐标转换：把全局坐标的目标点，转到车辆自己的局部坐标系下
dx = X_target - X;
dy = Y_target - Y;
x_local = dx * cos(phi) + dy * sin(phi);%局部坐标系下，车身前进方向为x轴正方向
y_local = -dx * sin(phi) + dy * cos(phi);%车身左侧是y轴正方向

  
    sigma = atan2(2 * lfr * y_local, Ld^2);%前轮转角sigma

    % ===============================================================

    % ================= TODO 2.2: 车辆状态更新 =================
    % 提示: 将刚才求得的转向角 sigma 代入运动学模型（复用第一问代码），更新 X, Y, phi。

phi_dot = v * tan(sigma) / lfr;
X = X + v * cos(phi) * dt;
Y = Y + v * sin(phi) * dt;
phi = phi + phi_dot * dt; 
    
    % ===============================================================
    
    % 到达终点提前结束
    if X >= X_ref(end), break; end
end

% 绘图对比
figure; hold on; grid on;
plot(X_ref, Y_ref, 'k--', 'LineWidth', 2);
plot(X_vec(1:ii), Y_vec(1:ii), 'r-', 'LineWidth', 2);
legend('参考规划轨迹', '实际行驶轨迹');
title(['Pure Pursuit 跟踪 (Ld = ', num2str(Ld), 'm)']);
xlabel('X [m]'); ylabel('Y [m]'); axis equal;