data = load("4_13_20A_0cm");
base = load("4_13_Base");

%Truncate data based off of initial plot, adjust values to delete flags
x_length = data(1,450:1279);
z_depth = data(2,450:1279);

%Original Data Plot
%plot(x_length, z_depth)
%figure;

%Trendline
basex = base(1,:);
basey = base(2,:);
coeff = polyfit(basex, basey, 1);
lin_fit = coeff(1)*x_length + coeff(2);

%Adjust Data for Surface Tilt
distance = z_depth - lin_fit;

%Filter data
filtered_z_depth = smoothdata(distance, "movmean", 25);

%Local Maxima
max = islocalmax(filtered_z_depth);

plot(x_length, distance);
hold on;
plot(x_length, filtered_z_depth, x_length(max),filtered_z_depth(max), 'r*');
title("4/13/25 20A 0am Depth vs Length");
xlabel("Length (mm)");
ylabel("Depth (mm)");
legend("Original Data", "Filtered Data", "Max Depth")