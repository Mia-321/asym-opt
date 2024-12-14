Implements of spectral GNNs are based on following repositories:

ChebNet, JacobiConv: https://github.com/GraphPKU/JacobiConv

GPRGNN: https://github.com/jianhao2016/GPRGNN

ChebNetII: https://github.com/ivam-he/ChebNetII

BernNet: https://github.com/ivam-he/BernNet



Conda environment can be:

conda env create -f asymopt.yml



To reproduce results on Texas, Wisconsin, Actor, Chameleon, Squirrel, Cornell, Citeseer, Pubmed, and Cora datasets, go to small directory;

To reproduce results on Roman_Empire, Amazon_Ratings, Minesweeper, Tolokers, Questions datasets, go to large-heter directory;

To reproduce results on Computers, Photo, Coauthor-CS, Coauthor-Physics datasets, go to large-homo directory;



An example of using ChebNet on Texas dataset (under small directory):

python3 -u training_asym.py --dataset Texas --net PolyNet --K 10 --lr1 5e-2 --lr2 5e-3 --lr3 1e-2 --wd1 1e-3 --wd2 5e-4 --wd3 5e-4 --a -0.5 --b 0.0 --dropout 0.6 --dprate 0.7 --base cheb --semi_rnd True --asym



# asym-opt
# asym-opt
