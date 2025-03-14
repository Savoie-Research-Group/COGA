copybottom(){
    nlines=$( grep -n TIME $1 | head -n 2 | tail -n 1 | awk -F":" '{ print $1 }' )
    nlines="$(( $nlines - 1 ))0"
    tail -n $nlines $1 > $2
}