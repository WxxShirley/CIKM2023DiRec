# Baseline Implementation: Recommender Models


## Intro 
We also consider item recommendation models for comparison. As CFAG does, we apply them on UGD task by treating each group as an item and thus only utilizing user-group interactions:

* MF-BPR: This is a classical recommendation model that
employs matrix factorization technique and is optimized by the
pair-wise BPR loss.
* NGCF (SIGIR'19): This method is a variant of standard GCN that
leverages high-order connectivity in a user-item bipartite graph.
* LightGCN (SIGIR'20): This is a method based on NGCF with optimization in training efficiency and generation ability by removing
feature transformation and non-linear activation.
* SGL (SIGIR'21): This work performs contrastive learning on LightGCN
model to augment node representations.
* SimGCL (SIGIR'22): This is a strong GNN-based recommender model
that incorporates an augmentation-free contrastive loss.
* DCCF (SIGIR'23): This is the state-of-the-art GNN-based recommender
model with disentangled representations.


We **unify all above methods** in this folder as in `model.py`.

## Reproducibility

To run these methods, you only need to specify the model type, for example:

```
# To run the SimGCL model
python main.py --dataset=Mafengwo --model_type=SimGCL
```

The running option `model_type` can only be **MF**, **NGCF**, **LightGCN**, **SGL**, **SimGCL**, and **DCCF**.

For more running options, please refer to `main.py`.

To reproduce the published results, please refer to `run.sh`.


## Acknowledgements

We thank the released official codes of existing baselines: LightGCN, SimGCL, and DCCF.

