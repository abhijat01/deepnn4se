# TODOs

## Move sections around 
The order in which material is presented needs to be changed. Vector gradients appear out of nowhere. 


## Backprop 
Backprop needs a more work. In particular, 

* start with single node and layer, show derivatives on it. 
* requires some image editing to show gradients on the bigger compute graph

## Layers and composition 
Since most of the people won't have a strong programming 
background, we should spend a little more time on the backprop, layers and how we can program the graph. 


## Compiling on command line 

 You might this usefull on windows with  MikTex and Texstudios installed in  portable mode: 

  ```
    MIKTEX_BASE=<i>
    "$MIKTEX_BASE/texmfs/install/miktex/bin/x64/pdflatex.exe" -synctex=1 -interaction=nonstopmode "slides".tex
  ```
