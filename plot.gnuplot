set xrange [0:7000]
set yrange [1e-16:1]
set logscale y
set key above maxrows 2 width -3
set xlabel "Frequency (GHz)"
set ylabel "Relative error"
f(x) = x**3/(exp(0.002448594197678852*x)-1) - x**3/(exp(0.017611906889726792*x)-1)
g(x) = exp(-(x/1500)**2)
noise_temp = 127. * pi/180 * 1e-26
noise_pol  = noise_temp * 2**0.5
set term svg size 650,500

set output "spec_error_rel_v4_lin_log_spin.svg"
plot "test10_nosub_uniform_00.txt" u 1:(abs($8/$2-1)) w l title "Uniform", for [i=1:2] "test13_a_mono.txt" u 1:(abs(column(4+i)/column(1+i)-1)) w l title sprintf("Spin-1 nowin %s", i==1?"T":i==2?"Q":"U"), for [i=1:2] "test13_c_mono.txt" u 1:(abs(column(4+i)/column(1+i)-1)) w l title sprintf("Spin-2 nowin %s", i==1?"T":i==2?"Q":"U"), (f(x)*g(x)+1e-15/1e-8)/(f(x)*g(x))-1 title "1e-15 noise" w l lw 2

set output "spec_error_rel_v4_lin_log.svg"
plot "test10_nosub_uniform_00.txt" u 1:(abs($8/$2-1)) w l title "Uniform", for [i=1:2] "test13_c_mono.txt" u 1:(abs(column(4+i)/column(1+i)-1)) w l title sprintf("No samp win %s", i==1?"T":i==2?"Q":"U"), for [i=1:2] "test13_sub51_c_mono.txt" u 1:(abs(column(4+i)/column(1+i)-1)) w l title sprintf("Samp win %s", i==1?"T":i==2?"Q":"U"), "" u 1:(abs(noise_temp/($2*g($1)))) title "Pixie noise T"  w l, "" u 1:(abs(noise_pol/($3*g($1)))) title "Pixie noise Q"  w l
set xrange [10:7000]
set logscale x
set output "spec_error_rel_v4_log_log.svg"
replot



set ylabel "Spectral flux density (Jy/sr)"
set yrange [1e-5:1e10]
set output "spec_error_abs_v4_log_log.svg"
plot for [i=1:2] "test13_sub51_c_mono.txt" u 1:(abs(column(4+i))*1e26) w l title sprintf("Recovered %s", i==1?"T":"Q"), for [i=1:2] "" u 1:(abs(column(4+i)-column(1+i))*1e26) w l title sprintf("Error in %s", i==1?"T":"Q"), "" u 1:(abs(noise_temp/g($1)*1e26)) title "Pixie noise T"  w l, "" u 1:(abs(noise_pol/g($1))*1e26) title "Pixie noise Q"  w l
set output "spec_error_abs_v4_log_log_nowindow.svg"
plot for [i=1:2] "test13_c_mono.txt" u 1:(abs(column(4+i))*1e26) w l title sprintf("Recovered %s", i==1?"T":"Q"), for [i=1:2] "" u 1:(abs(column(4+i)-column(1+i))*1e26) w l title sprintf("Error in %s", i==1?"T":"Q"), "" u 1:(abs(noise_temp/g($1)*1e26)) title "Pixie noise T"  w l, "" u 1:(abs(noise_pol/g($1))*1e26) title "Pixie noise Q"  w l

unset logscale x
set xrange [0:7000]
set output "spec_error_abs_v4_lin_log.svg"
plot for [i=1:2] "test13_sub51_c_mono.txt" u 1:(abs(column(4+i))*1e26) w l title sprintf("Recovered %s", i==1?"T":"Q"), for [i=1:2] "" u 1:(abs(column(4+i)-column(1+i))*1e26) w l title sprintf("Error in %s", i==1?"T":"Q"), "" u 1:(abs(noise_temp/g($1)*1e26)) title "Pixie noise T"  w l, "" u 1:(abs(noise_pol/g($1))*1e26) title "Pixie noise Q"  w l
set output "spec_error_abs_v4_lin_log_nowindow.svg"
plot for [i=1:2] "test13_c_mono.txt" u 1:(abs(column(4+i))*1e26) w l title sprintf("Recovered %s", i==1?"T":"Q"), for [i=1:2] "" u 1:(abs(column(4+i)-column(1+i))*1e26) w l title sprintf("Error in %s", i==1?"T":"Q"), "" u 1:(abs(noise_temp/g($1)*1e26)) title "Pixie noise T"  w l, "" u 1:(abs(noise_pol/g($1))*1e26) title "Pixie noise Q"  w l

unset logscale y
set yrange [-5e5:4e6]
set xrange [0:5000]
set output "spec_error_abs_v4_lin_lin.svg"
plot for [i=1:2] "test13_sub51_c_mono.txt" u 1:(column(4+i)*1e26) w l title sprintf("Recovered %s",i==1?"T":"Q"), for [i=1:2] "test13_sub51_c_mono.txt" u 1:((column(4+i)-column(1+i))*1e26*1e5) w l title sprintf("Error in %s*1e5",i==1?"T":"Q"), "" u 1:(abs(noise_temp/g($1)*1e26*1e4)) title "Pixie noise T*1e4"  w l, "" u 1:(abs(noise_pol/g($1))*1e26*1e4) title "Pixie noise Q*1e4"  w l
set output "spec_error_abs_v4_lin_lin_nowindow.svg"
plot for [i=1:2] "test13_c_mono.txt" u 1:(column(4+i)*1e26) w l title sprintf("Recovered %s",i==1?"T":"Q"), for [i=1:2] "test13_c_mono.txt" u 1:((column(4+i)-column(1+i))*1e26*1e7) w l title sprintf("Error in %s*1e7",i==1?"T":"Q"), "" u 1:(abs(noise_temp/g($1)*1e26*1e4)) title "Pixie noise T*1e4"  w l, "" u 1:(abs(noise_pol/g($1))*1e26*1e4) title "Pixie noise Q*1e4"  w l
