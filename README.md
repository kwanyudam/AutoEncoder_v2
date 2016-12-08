# KKY_autoencoder

Kwan Yu Kim AutoEncoder

Requirements  
---
tensorflow python

How to Use
---
- prepare an input file
~~~
15
filename	hips	neck	lsho	lelb	lhand	rsho	relb	rhand	lfem	lknee	lank	rfem	rknee	rank
filename>0 0 0 1 2 3 5 6 ...
filename>0 0 0 1 2 3 5 6 ...
.
.
.
~~~
- in terminal >python preprocess.py [Network Name] [Input Filename] [Output Filename]  
ex) python preprocess.py network.ckpt input_db.bin output_db.bin
- Output file example
~~~
0 0 0 1 2 3 5 6 ...
0 0 0 1 2 3 5 6 ...
.
.
.
~~~
