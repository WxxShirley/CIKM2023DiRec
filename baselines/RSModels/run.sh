DATASET=Mafengwo
DEVICE=cuda:0

for MODEL_TYPE in DCCF NGCF LightGCN SGL SimGCL MF; do
   python -u main.py --dataset=$DATASET --device=$DEVICE --model_type=$MODEL_TYPE >logs/$DATASET/$MODEL_TYPE.log
done


DATASET=Weeplaces
for MODEL_TYPE in DCCF NGCF LightGCN SGL SimGCL MF; do
   python -u main.py --dataset=$DATASET --device=$DEVICE --model_type=$MODEL_TYPE >logs/$DATASET/$MODEL_TYPE.log
done


DATASET=Steam
for MODEL_TYPE in DCCF NGCF LightGCN SGL SimGCL MF; do
   python -u main.py --dataset=$DATASET --device=$DEVICE --model_type=$MODEL_TYPE --num_negatives=4 --epoch=100 >logs/$DATASET/$MODEL_TYPE.log
done
