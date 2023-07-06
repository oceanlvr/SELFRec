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


## LightGCN

benchmark on yelp2018 / iFashion / douban-book / amazon-kindle


```sh
nohup python index.py --gpu_id=0 --group benchmark --job_type train --run_name LightGCN_1 --model=LightGCN --dataset=yelp2018 > ./0.log 2>&1 &
nohup python index.py --gpu_id=1 --group benchmark --job_type train --run_name LightGCN_2 --model=LightGCN --dataset=iFashion --num_epochs=120 > ./1.log 2>&1 &
nohup python index.py --gpu_id=2 --group benchmark --job_type train --run_name LightGCN_3 --model=LightGCN --dataset="douban-book" --num_epochs=120 > ./2.log 2>&1 &
nohup python index.py --gpu_id=3 --group benchmark --job_type train --run_name LightGCN_4 --model=LightGCN --dataset="amazon-kindle" --num_epochs=120 > ./3.log 2>&1 &
```


## SGL

benchmark on yelp2018 / iFashion / douban-book / amazon-kindle

```sh
nohup python index.py --gpu_id=0 --group benchmark --job_type train --run_name SGL_1 --model=SGL --dataset=yelp2018 > ./0.log 2>&1 &
nohup python index.py --gpu_id=1 --group benchmark --job_type train --run_name SGL_2 --model=SGL --dataset=iFashion --num_epochs=120 > ./1.log 2>&1 &
nohup python index.py --gpu_id=2 --group benchmark --job_type train --run_name SGL_3 --model=SGL --dataset="douban-book" --num_epochs=120 > ./2.log 2>&1 &
nohup python index.py --gpu_id=3 --group benchmark --job_type train --run_name SGL_4 --model=SGL --dataset="amazon-kindle" --num_epochs=120 > ./3.log 2>&1 &
```


## NCL

benchmark on yelp2018 / iFashion / douban-book / amazon-kindle

```sh
nohup python index.py --gpu_id=0 --group benchmark --job_type train --run_name NCL_1 --model=NCL --dataset=yelp2018 --batch_size=1024  > ./0.log 2>&1 &
nohup python index.py --gpu_id=1 --group benchmark --job_type train --run_name NCL_2 --model=NCL --dataset=iFashion --num_epochs=120 --batch_size=1024 > ./1.log 2>&1 &
nohup python index.py --gpu_id=2 --group benchmark --job_type train --run_name NCL_3 --model=NCL --dataset="douban-book" --num_epochs=120 --batch_size=1024  > ./2.log 2>&1 &
nohup python index.py --gpu_id=3 --group benchmark --job_type train --run_name NCL_4 --model=NCL --dataset="amazon-kindle" --num_epochs=120 --batch_size=1024  > ./3.log 2>&1 &
```
