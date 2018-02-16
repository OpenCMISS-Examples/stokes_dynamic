$dt=1;                         # frequency
$steps=40000;                  # number of time steps
$cores=1;                      # number of processors

$w=500;                        # width of the graphical window
$h=1000;                       # height of the graphical window

#Read in the sequence of nodal positions.
for ($i=0;$i<$steps;$i=$i+$dt)
  {
     $time = $i/($dt*100)
     for ($j=0;$j<$cores;$j=$j+1)
       {
          $filename = sprintf("./output/MainTime_%01d.part%01d.exnode", $i, $j);
          print "Reading $filename time $time\n";
          gfx read node "$filename" time $i;
       }
  }

#Read in the element description
for ($k=0;$k<$cores;$k=$k+1)
  {
     $filename = sprintf("./output/MainTime_0.part%01d.exelem", $k);
     gfx read element "$filename";
  }

gfx define faces egroup StokesRegion

gfx define field Coordinate.x component Coordinate.x
gfx define field Coordinate.y component Coordinate.y
gfx define field Coordinate.z component Coordinate.z

gfx def field x_velocity component U.1
gfx def field y_velocity component U.2
gfx def field z_velocity component U.3
gfx def field pressure component U.4

gfx define field vector_field coord rectangular_cartesian component U.1 U.2 U.3 U.4
gfx modify g_element StokesRegion node_points data y_velocity

gfx edit scene
gfx create window 1;

#Set the timekeeper playing
gfx timekeeper default set 1.0;
gfx timekeeper default speed 1;
gfx create time_editor
