figure;
data = load("2_9_D2.txt");

%Truncate data based off of initial plot, adjust values to delete flags
x_length = data(1,1:1241);
z_depth = data(2,1:1241);

%Original Data Plot
%plot(x_length, z_depth)
%figure;

%Trendline
z_linear = z_depth;
x_linear = x_length;
coeff = polyfit(x_linear, z_linear, 1);
lin_fit = coeff(1)*x_length + coeff(2);

%Adjust Data for Surface Tilt
distance = z_depth - lin_fit;

%Filter data
filtered_z_depth = smoothdata(distance, "movmean", 15);

%Local Maxima
max = islocalmax(filtered_z_depth);

plot(x_length, distance);
hold on;
plot(x_length, filtered_z_depth, x_length(max),filtered_z_depth(max), 'r*');
title("2/9 A1 Depth vs Length");
xlabel("Length (mm)");
ylabel("Depth (mm)");
legend("Original Data", "Filtered Data", "Max Depth")