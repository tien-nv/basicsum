import nltk
import os
import rouge
# import nltk
from shutil import copyfile
import statistics as sta
import sys
sys.setrecursionlimit(15000)
# nltk.download('punkt')
def prepare_results(metric,p, r, f):
    # return '\t{}:\t{}: {:5.2f}'.format(metric, 'F1', 100.0 * f)
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'Percision', 100.0 * p, 'Recall', 100.0 * r, 'F1-score', 100.0 * f)


def my_eval():
    hyp = 'outputs/hyp'
    # ref = 'summary_to_evaluate'
    raw_ref = 'outputs/ref'    
    FJoin = os.path.join
    files_hyp = [FJoin(hyp, f) for f in os.listdir(hyp)]
    files_raw_ref = [FJoin(raw_ref, f) for f in os.listdir(raw_ref)]
    
    f_hyp = []
    f_raw_ref = []
    for file in files_hyp:
        f = open(file,encoding='utf-8')
        f_hyp.append(f.read())
        f.close()
    for file in files_raw_ref:
        f = open(file,encoding='utf-8')
        f_raw_ref.append(f.read())
        f.close()
    print("compute 75bytes: ")
    mrouge_75 = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],max_n=2,
                             limit_length=True,length_limit=75,
                             length_limit_type='bytes',apply_avg=True)
    scores = mrouge_75.get_scores(f_hyp,f_raw_ref)
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
          print(prepare_results(metric,results['p'], results['r'], results['f']))
    print("compute 275bytes: ")
    mrouge_275 = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],max_n=2,
                             limit_length=True,length_limit=275,
                             length_limit_type='bytes',apply_avg=True)
    scores = mrouge_275.get_scores(f_hyp,f_raw_ref)
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
          print(prepare_results(metric,results['p'], results['r'], results['f']))
    print("compute full lenght: ")
    mrouge_full = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],max_n=2,
                             limit_length=False,
                             length_limit_type='bytes',apply_avg=True)
    scores = mrouge_full.get_scores(f_hyp,f_raw_ref)
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
          print(prepare_results(metric,results['p'], results['r'], results['f']))

if __name__ == '__main__':
    my_eval()