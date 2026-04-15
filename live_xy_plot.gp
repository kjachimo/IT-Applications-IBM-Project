# Live XY plot from a growing stream file.
#
# Usage:
#   gnuplot -e "datafile='stream.xy'; refresh=0.2; plottitle='Live XY Stream'" live_xy_plot.gp
#
# Expected data format (appended over time):
#   x y
#   0.0 1.2
#   0.1 1.3

if (!exists("datafile")) datafile = "stream.xy"
if (!exists("refresh"))  refresh  = 0.2
if (!exists("plottitle")) plottitle = "Live XY Stream"

set term qt 
set title plottitle
set xlabel "x"
set ylabel "y"
set grid
set key left top

while (1) {
    plot datafile using 1:2 with linespoints lw 2 pt 7 ps 0.5 title "stream"
    pause refresh
}
