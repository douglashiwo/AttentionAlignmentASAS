from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
import time, os, random, bs4
from transformers import AdamW
from transformers import get_scheduler
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import roc_curve, roc_auc_score, auc,precision_score, f1_score, accuracy_score, recall_score
from datasets import Dataset
import argparse
import time
import scipy.stats as ss
import ast
import ml_metrics as metrics


def input_to_attention_distribution(toks, result,simple_res = True):
    'score_type: answer2answer, refer2answer, both'
    def find_answer_end(lst):
        ix = 0
        length = len(lst)
        while ix < length:
            if lst[ix] == 0:
                return ix - 2
            ix += 1
        # all are one, then ix=length, we return length-2   
        return length - 2
    
    # result means model output for example, the output of BERT.
    '''
    input_batch = toks.input_ids.tolist()
    token_type_batch = toks.token_type_ids.tolist()
    attention_mask_batch = toks.attention_mask.tolist()
    attention_batch = result.attentions[-1].tolist()
    attention_score_batch=[]    
    '''
    input_batch = toks['input_ids'].tolist()
    token_type_batch = toks['token_type_ids'].tolist()
    attention_mask_batch = toks['attention_mask'].tolist()
    attention_batch = result.attentions[-1].tolist()
    attention_score_batch=[]  
      
    for ix, input_ids in enumerate(input_batch):
        answer_start = 1
        
        answer_end = find_answer_end(attention_mask_batch[ix])   #.index(1) - 2
        #answer_end = token_type_batch[ix].index(1) - 2
        #referanswer_start = token_type_batch[ix].index(1)
        #referanswer_end = find_referanswer_end(attention_mask_batch[ix])
        answer_length = answer_end - answer_start + 1
        #referanswer_length = referanswer_end - referanswer_start+1

        score_dict=[]

        atn = np.mean(np.array(attention_batch[ix]), axis=0) #average over all 12 heads
        for i, tk in enumerate(input_ids):
            if i>=answer_start and i<= answer_end:# and tokenizer.decode(tk) not in (',','.','?','!'):
                
                aten_score_for_i = np.sum(atn[answer_start : answer_end + 1, i])*1.0 / answer_length
                if tokenizer.decode(tk) in (',','.','?','!',':',"'",'"'):
                    aten_score_for_i = 0.0
                score_dict.append((str(i)+'_'+str(tk)+"_"+tokenizer.decode(tk), round(aten_score_for_i,6)))
                
                #aten_score_for_i_a = np.sum(atn[answer_start : answer_end + 1, i])*1.0 / answer_length
                #aten_score_for_i_b = np.sum(atn[referanswer_start : referanswer_end + 1, i])*1.0 / referanswer_length
                #aten_score_for_i_ab = aten_score_for_i_a + aten_score_for_i_b
                #score_dict.append((str(i)+'_'+str(tk)+"_"+tokenizer.decode(tk), str(round(aten_score_for_i_a,6))+"_"+str(round(aten_score_for_i_b,6))+'_'+str(round(aten_score_for_i_ab,6))))

        attention_score_batch.append(score_dict)
                 
    recovered_res = []
    
    for tar in attention_score_batch:

        recover_list = []
        l = len(tar)

        for ix, tup in enumerate(tar):
            if ix == 0:
                recover_list.append([tup])
            elif tar[ix-1][0].split('_')[2].startswith('#') is False and tar[ix][0].split('_')[2].startswith('#') is False:
                #print('we are append: ',[tup])
                recover_list.append([tup])
            elif tar[ix-1][0].split('_')[2].startswith('#') is False and tar[ix][0].split('_')[2].startswith('#') is True:
                recover_list[-1].append(tup)
            elif tar[ix-1][0].split('_')[2].startswith('#') is True and tar[ix][0].split('_')[2].startswith('#') is True:
                recover_list[-1].append(tup)        
            elif tar[ix-1][0].split('_')[2].startswith('#') is True and tar[ix][0].split('_')[2].startswith('#') is False:
                #print('we are append: ',[tup])
                recover_list.append([tup])

        res = []
        for gp in recover_list:
            recovered_word = (tokenizer.decode([int(  e[0].split('_')[1]  ) for e in gp]))
            word_attention_score = np.sum([e[1] for e in gp])
            res.append((recovered_word, word_attention_score))       

        recovered_res.append(res) 
               
    if simple_res is True:
        return recovered_res    
    return attention_score_batch, recovered_res


def attention_rank(attention_scores):  #checked
    attention_scores1 = [e[0] for e in attention_scores]
    attention_scores2 = [round(e[1],6) for e in attention_scores]
    attention_scores3 = ss.rankdata([1.0/(e+0.00001) for e in attention_scores2 ])
    attention_scores3 = [int(e) for e in attention_scores3]
    return list(zip(attention_scores1,attention_scores2,attention_scores3))


def attention_score(ans, model):
    
    toks = tokenizer(ans, padding='max_length', return_tensors="pt") # model and input should be transferred to cuda before forwarding
    
    toks = {k: v.to(device) for k, v in toks.items()}
    
    result = model(**toks,
                   output_attentions=True,
                   output_hidden_states=False )
    
    predictions = torch.argmax(result.logits, dim =- 1).view(-1).tolist()
    
    scores = input_to_attention_distribution(toks, result)
    
    scores = [attention_rank(e) for e in scores]
    
    return scores, predictions


if __name__ == "__main__":   #checked

    parser = argparse.ArgumentParser(description="This is a description")
       
    parser.add_argument('--output_file',dest='output_file',required = True,type = str)
    parser.add_argument('--model_file',dest='model_file',required = True,type = str)
    parser.add_argument('--test_file',dest='test_file',required = True,type = str)
    
    args = parser.parse_args()    
    
    output_file = args.output_file
    model_file = args.model_file
    test_file = args.test_file
       
    print('output_file: {}'.format(output_file)) 
    print('model_file: {}'.format(model_file))
    print('test_file: {}'.format(test_file))
    
    t1 = time.time()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")    
    model = torch.load(model_file).to(device)

    test = pd.read_excel(test_file)
    ans = test['studentanswer']
    
    tpl = list(ans)

    res = []
    
    predictions = []
    
    for ans in tpl:
        #print(ans,refer)
        scores, pred = attention_score(ans, model)#, score_type = 'both') # score_type = 'answer2answer' 'both'  'refer2answer'

        res.append(scores[0]) #[0] for unsqueeze because we donnot input batch actually
        
        predictions = predictions + pred       
    
    test['atn_score'] = pd.Series(res)

    test['predictions'] = pd.Series(predictions)
    
    test.to_excel(output_file,index=False)
    
    print('QWK value between human grader and model: ',round(metrics.quadratic_weighted_kappa(test['score1'],test['predictions']),4))
    print('QWK value between two human grader is : ',round(metrics.quadratic_weighted_kappa(test['score1'],test['score2']),4))
    print('RMSE value between two human grader is : ',round(metrics.rmse(test['score1'],test['score2']),4))
    print('RMSE value between human grader and model : ',round(metrics.rmse(test['score1'],test['predictions']),4))
        
    print('time cost: {} secs'.format(int(time.time()-t1)))  