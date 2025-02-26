# LSEnet

LSEnet: Lorentz Structural Entropy Neural Network for Deep Graph Clustering.

## Get Started

```bash
cd ./LSEnet
python main.py
```

## Visualization

<div align=center>
<img src="./images/FootBall_pred.png" width=50% alt="football" title="FootBall" >
</div>
<div align=center>
Figure 1. Prediction results on FootBall dataset.
</div>
<br><br>
<div align=center>
<img src="./images/FootBall_true.png" width=50% alt="football" title="FootBall">
</div>
<div align=center>
Figure 2. True labels of FootBall dataset.
</div>
# deep_graph_clustering
sbatch --gres=gpu:a100:1 -C a100_80 --time=24:00:00 --output=res_lsenet.txt fau_alex_job_script.sh
