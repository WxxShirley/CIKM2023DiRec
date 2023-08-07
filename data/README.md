## Datasets


We use the raw data released in CFAG ([repo](https://github.com/mdyfrank/CFAG)).
We **re-split** the dataset into train set, validation set, and test set with the ratios of 70%, 10%, and 20%, respectively.

### Data Statistics 

| Dataset |Mafengwo | Weeplaces | Steam   |
| ------- | ------- | ----------| ------- |
| # Users | 1,269  | 1,501 | 11,099 |
| # Groups| 972    | 4,651 | 1,085  |
| # Items | 999    | 6,406 | 2,351  |
| # User-Group Participation | 5,574 | 12,258 | 57,654 |
| # User-Item Interactions | 8,676 | 43,942 | 444,776 |
| # Group-Item Interactions | 2,540 | 6,033 | 22,318 |


### Data format

In each folder, there are 5 files:

* `groupItemTrain.txt`: This file contains **user-item interactions**. In each line, the format is:
   
   ```
   user_id, interacted_item_id1, interacted_item_id2, ...
   ```

* `userItemTrain.txt`: This file contains **group-item interactions**. In each line, the format is:
  
  ``` 
  group_id, interacted_item_id1, interacted_item_id2, ...
  ```

* `train.txt`: This file contains **training data of user-group participation**. In each line, the format is:
  
  ```
  user_id, joined_group_id1, joined_group_id2, ...
  ```

* `val.txt`: This file contains validation data of user-group participation.

* `test.txt`: This file contains test data of user-group participation.

