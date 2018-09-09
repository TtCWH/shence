# -*- coding:utf-8 -*-
__author__ = 'Han Wang'

import pdb
import jieba
import re
import argparse
import jieba.analyse
import jieba.posseg
import numpy as np
import pandas as pd


def add_arguments(parser):
    parser.add_argument("--mode", type=str, default='train', help="test on trainSet or testSet")
    parser.add_argument("--topK", type=int, default=10, help="arg in jieba.analyse.extract_tags")
    parser.add_argument("--subNumber", type=bool, default=False, help="sub the number in title")
    parser.add_argument("--subBookTitle", type=bool, default=True, help="sub the 《》 in title")
    parser.add_argument("--use_userDict", type=bool, default=True, help="load_userdict")
    parser.add_argument("--all_file", type=str, default='./data/all_docs.txt', help="file path of all_docs")
    parser.add_argument("--train_file", type=str, default='./data/train_docs_keywords.txt', help="file path of train_docs")
    parser.add_argument("--word_set", type=str, default='./data/res_words1000.txt', help="file path of word_set")
    parser.add_argument("--userDict", type=str, default='./data/words.txt', help="file path of userdict")


def read_file(args):
    data = pd.read_csv(args.all_file, sep='\001', header=None)
    data.columns = ['id','title','doc']

    train = pd.read_csv(args.train_file, sep='\t', header=None)
    train.columns = ['id','label']
    train_id_list = list(train['id'].unique())

    train_title_doc = data[data['id'].isin(train_id_list)]

    test_title_doc = data[~data['id'].isin(train_id_list)]

    train_title_doc = pd.merge(train_title_doc,train,on=['id'],how='inner')

    return train_title_doc, test_title_doc


def get_it(args):
    ret={}
    p=0
    with open(args.word_set) as f:
        for _ in f.readlines():
            ret[_.split()[0]]=p
            p+=1
    return ret


def get_dic(data,dic):
    for elem in data['title']:
        words=[x.word for x in jieba.posseg.cut(elem)]
        flags=[x.flag for x in jieba.posseg.cut(elem)]
        dic.update(dict(zip(words,flags)))


def main():
    parser=argparse.ArgumentParser()
    add_arguments(parser)
    args=parser.parse_args()
    if args.use_userDict:
        jieba.load_userdict(args.userDict)
    it=get_it(args)
    train_set,test_set=read_file(args)
    if args.mode=='train':
        data=train_set
    else:
        data=test_set

    # 提取书名号的内容
    data['title_re'] = data['title'].apply(lambda x: ','.join(re.findall(r"《(.+?)》", str(x))))

    if args.subNumber:
        data['title'] = data['title'].apply(lambda x: ''.join(filter(lambda ch: ch not in ' \t1234567890', str(x))))

    if args.subBookTitle:
        data['title'] = data['title'].apply(lambda x: re.sub(r"《(.+?)》", '', str(x)))

    dic={} # 记录所有词的词性
    #data['title_cut'] = data['title_cut'].apply(lambda x:''.join(filter(lambda ch: ch not in ' \t1234567890，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥', str(x))))
    get_dic(data,dic)

    data['title_word_pos']=data['title'].apply(lambda x: ','.join(["{}/{}".format(x.word,x.flag) for x in jieba.posseg.cut(str(x))]))

    # 用tfidf提取关键词
    data['title_tfidf'] = data['title'].apply(lambda x:','.join(jieba.analyse.extract_tags(str(x),topK = args.topK)))

    data_result = data[['id', 'title', 'title_word_pos', 'title_tfidf', 'title_re']]
    if args.mode == 'train':
        data_result = data[['id', 'label', 'title', 'title_word_pos', 'title_tfidf', 'title_re']]

    label1 = []
    label2 = []

    cnt=0

    for elem in data_result.values:
        if args.mode=='train':
            gold = str(elem[1]).split(',')
        title_tfidf = str(elem[-2]).split(',')
        title_re = str(elem[-1]).split(',')
        p=0
        if title_re[0] != '':
            tmp_result=title_re
        else:
            tmp_result=[]
        while len(tmp_result)<2 and p<len(it.keys()):
            for _ in title_tfidf:
                for tmp in tmp_result:
                    if _ in tmp:
                        continue
                if not dic.get(_):
                    continue
                if not it.get(dic[_]):
                    continue
                if it[dic[_]]==p:
                    tmp_result.append(_)
            p+=1

        if args.mode=='train':
            cnt+=len(set(gold)&set(tmp_result[:2]))
        if len(tmp_result) > 1:
            label1.append(tmp_result[0])
            label2.append(tmp_result[1])
        elif len(tmp_result) == 1:
            label1.append(tmp_result[0])
            label2.append(tmp_result[0])
        else:
            label1.append('')
            label2.append('')

    if args.mode=='train':
        print(cnt*0.5)
    result = pd.DataFrame()

    ids = data_result['id'].unique()

    result['id'] = list(ids)
    if args.mode=='train':
        result['gold'] = data_result['label']
    result['label1'] = label1
    result['label1'] = result['label1'].replace('','nan')
    result['label2'] = label2
    result['label2'] = result['label2'].replace('','nan')
    result['title_word_pos'] = data_result['title_word_pos']
    result['title'] = data_result['title']

    print('done')
    if args.use_userDict:
        arg0='useDict'
    else:
        arg0='useNone'
    result.to_csv('./result/{}_{}_topK{}'.format(args.mode,arg0,args.topK),index=None)
    if args.mode=='test':
        result[['id'],'label1','label2'].to_csv('./result/upload_{}_{}_topK{}'.format(args.mode,arg0,args.topK),index=None)


if __name__ == '__main__':
    main()

