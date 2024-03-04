from src.constants import *

def confusion_matrix(pred_labels, true_labels, positive_class, negative_class):
    
    true_positive=0
    false_positive=0
    false_negative=0
    true_negative=0
    
    for pred, true in zip(pred_labels, true_labels):
        
        if pred==positive_class and true==positive_class:
            true_positive+=1
            
        if pred==positive_class and true==negative_class:
            false_positive+=1
            
        if pred==negative_class and true==positive_class:
            false_negative+=1
            
        if pred==negative_class and true==negative_class:
            true_negative+=1
            
    return true_positive, false_positive, false_negative, true_negative


def accuracy(predicted, actual):
    true_positive, false_positive, false_negative, true_negative=confusion_matrix(predicted, actual,POSITIVE_CLASS, NEGATIVE_CLASS)
    acc=(true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
    return round(acc*100, 2)

def precision(predicted, actual):
    true_positive, false_positive, false_negative, true_negative=confusion_matrix(predicted, actual,POSITIVE_CLASS, NEGATIVE_CLASS)
    prec=(true_positive)/(true_positive + false_positive)
    return prec

def recall(predicted, actual):
    true_positive, false_positive, false_negative, true_negative=confusion_matrix(predicted, actual, POSITIVE_CLASS, NEGATIVE_CLASS)
    rec=(true_positive)/(true_positive + false_negative)
    return rec

def specifity(predicted, actual):
    true_positive, false_positive, false_negative, true_negative=confusion_matrix(predicted, actual,POSITIVE_CLASS, NEGATIVE_CLASS)
    specifity=(true_negative)/(true_negative + false_positive)
    return specifity

def f1_score(predicted, actual):
    true_positive, false_positive, false_negative, true_negative=confusion_matrix(predicted, actual,POSITIVE_CLASS, NEGATIVE_CLASS)
    prec=(true_positive)/(true_positive + false_positive)
    rec=(true_positive)/(true_positive + false_negative)
    score=(2*prec*rec)/(prec+rec)
    return score