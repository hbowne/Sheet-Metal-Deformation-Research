data = load("4_13_10A_1cm");
base = load("4_13_Base");

%Original Scan is already relatively flat

%Truncate data based off of initial plot, adjust values to delete flags
x_length = data(1,230:1000);
z_depth = data(2,230:1000);

%Original Data Plot
%plot(x_length, z_depth)
%figure;

%Trendline
%basex = base(1,:);
%basey = base(2,:);
%coeff = polyfit(basex, basey, 1);
%lin_fit = coeff(1)*x_length + coeff(2);

%Adjust Data for Surface Tilt
%distance = z_depth - lin_fit;

%Filter data
filtered_z_depth = smoothdata(z_depth, "movmean", 30);

%Local Maxima
max = islocalmax(filtered_z_depth);

plot(x_length, z_depth);
hold on;
plot(x_length, filtered_z_depth, x_length(max),filtered_z_depth(max), 'r*');
title("4/13/25 10A 1cm Depth vs Length");
xlabel("Length (mm)");
ylabel("Depth (mm)");
legend("Original Data", "Filtered Data", "Max Depth")