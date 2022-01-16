
# This script is used for training the automatic short answer grading model for the processed dataset (one can find the data from the ./data directory).
# Here are a exampleto train a model with the following hyperparamether:
# Source code file : code.py (This model is based on the Transformers library, adopting the pre-trained Bert model for sequence classification). Refer to https://github.com/huggingface/transformers for more details.


# Learning rate: 0.00005
# Actual batch_size: 16 (Actual batch_size = batch_size * accumulation_steps)
# Number of num_epochs: 5 
# Num_cls: number of total classess in the dataset(training,valid and test AS a whole)
# Training file: data/essay_sub_train1.xlsx
# Valid file: data/essay_sub_valid1.xlsx
# Test file: data/essay_sub_test1.xlsx
# Output model file: best_bert_lr00005_sub_1.bert (the best model is automatically saved (under QWK metric))
# Finally, some model details will be recorded in "result/lr00005_sub_1.txt".

python -u code.py  --train_file data/essay_sub_train1.xlsx  --valid_file data/essay_sub_valid1.xlsx --test_file data/essay_sub_test1.xlsx  --accumulation_steps 16 --num_epochs 5 --batch_size 1 --learning_rate 0.00005  --best_model best_bert_lr00005_sub_1.bert  --num_cls 4  > result/lr00005_sub_1.txt
