#FILE="stream2.txt"
set output "stream2.png"
set terminal png
set logscale x
set xlabel '#elements in vector'
set ylabel 'MB/s'
plot FILE using 1:3 title "FILL" with linespoints, FILE using 1:4 title "COPY" with linespoints, FILE using 1:5 title "DAXPY" with linespoints, FILE using 1:6 title "DOT" with linespoints
#pause -1
