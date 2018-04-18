x = -5:10;
y = -5:10;
[xx, yy] = meshgrid(x, y);		% xx 和 yy 都是矩陣  
zz = 1/pi/sqrt(3/4)*exp(-1/2*(4/3*xx.^2+16/3*yy.^2+8/3*xx.*yy-8*xx-16*yy+16));				% 計算函數值 zz，也是矩陣
figure;

mesh(xx,yy,zz); %draw mesh
hold on;
contour(xx,yy,zz,[0.01,0.01],'ShowText','on');
title('pdf'); 
xlabel('x');
ylabel('y');
hold off;