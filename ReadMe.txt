
# ----------------------------------------For script.bat------------------------------------------
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





# ----------------------------------------For atn-score.bat------------------------------------------

# This script is used for assigning attention score for each token within the input sentence.
# Before using this script, one should first train a model (obtain the saved model file ending with xxx.bert) using the script.bat.


# Source code file: attention-calculation.py
# Test file (containing the text sentence) to be highlighted: data/essay_sub_test1.xlsx
# Output file (with attention score): aten-test_sub_1.xlsx
# Model being used: best_bert_lr00005_sub_1.bert


# Finally, some running info details will be recorded in "aten_sub_test_1.txt".