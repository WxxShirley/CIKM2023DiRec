# Group Recommendation Baselines

Group Recommendation (GR) task is defined as suggesting suitable items for groups. 
To empirically show the inferior performance of applying GR on UGD task, we consider the following GR methods. 

Note that to apply these methods to UGD task, we adapt them by replacing their initial *group-item BPR loss* with **user-group BPR loss**:

* **AGRE** (SIGIR'18): This is a classical group recommendation method.

* **ConsRec** (WWW'23): This is the state-of-the-art group recommendation method.


