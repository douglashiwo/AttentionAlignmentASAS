# This script is used for assigning attention score for each token within the input sentence.
# Before using this script, one should first train a model (obtain the saved model file ending with xxx.bert) using the script.bat.


# Source code file: attention-calculation.py
# Test file (containing the text sentence) to be highlighted: data/essay_sub_test1.xlsx
# Output file (with attention score): aten-test_sub_1.xlsx
# Model being used: best_bert_lr00005_sub_1.bert


# Finally, some running info details will be recorded in "aten_sub_test_1.txt".


python -u attention-calculation.py  --test_file data/essay_sub_test1.xlsx  --output_file aten-test_sub_1.xlsx  --model_file best_bert_lr00005_sub_1.bert > aten_sub_test_1.txt
