# LeaderBoard

douban-book / iFashion / yelp2018

<h2>Leaderboard</h2>
The results are obtained on the dataset of <b>Yelp2018</b>. We performed grid search for the best hyperparameters. <br>
General hyperparameter settings are: batch_size: 2048, emb_size: 64, learning rate: 0.001, L2 reg: 0.0001. <br><br>


|  Model   |      Recall@20      | NDCG@20 | Hyperparameter settings                                                                             |
|:--------:|:-------------------:|:-------:|:----------------------------------------------------------------------------------------------------|
|   LightGCN    |       0.0639        | 0.0525  |     layer=3     |
|   NCL    |       0.0670        | 0.0562  | layer=3, ssl_reg=1e-6, proto_reg=1e-7, tau=0.05, hyper_layers=1, alpha=1.5, num_clusters=2000 |
|   SGL    |       0.0675        | 0.0555  |     λ=0.1, ρ=0.1, tau=0.2 layer=3     |
|  SimGCL  |       0.0721        | 0.0601  |   λ=0.5, eps=0.1, tau=0.2, layer=3    |
| XSimGCL  |       0.0723        | 0.0604  | λ=0.2, eps=0.2, l∗=1 tau=0.15 layer=3 |
