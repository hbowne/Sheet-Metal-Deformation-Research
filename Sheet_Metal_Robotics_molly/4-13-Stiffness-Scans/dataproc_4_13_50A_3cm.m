%Scan already level
figure;
data = load("4_13_50A_3cm");
dataAve = load("4_13_base");

coeff = polyfit(dataAve(1, 1:1200), dataAve(2, 1:1200), 1);
%plot(dataAve(1, 1:1200), dataAve(2, 1:1200))

%Truncate data based off of initial plot, adjust values to delete flags
x_length = data(1,457:1245);
z_depth = data(2,457:1245);

%Original Data Plot
%plot(x_length, z_depth)
%figure;

%Trendline
z_linear = z_depth;
x_linear = x_length;
%coeff = polyfit(x_linear, z_linear, 1);
lin_fit = coeff(1)*x_length + coeff(2);

%Adjust Data for Surface Tilt
distance = z_depth;% - lin_fit;

%Filter data
filtered_z_depth = smoothdata(distance, "movmean", 15);

%Local Maxima
max = islocalmax(filtered_z_depth);

plot(x_length, distance);
hold on;
plot(x_length, filtered_z_depth, x_length(max),filtered_z_depth(max), 'r*');
title("4/13 50A 3cm Depth vs Length");
xlabel("Length (mm)");
ylabel("Depth (mm)");
legend("Original Data", "Filtered Data", "Max Depth")
