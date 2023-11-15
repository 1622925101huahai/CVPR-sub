import pickle
from operator import itemgetter

import numpy
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import RobertaTokenizer
from codes.DATASET import DATASET
from codes.Function import modelWightsDir, flitIllWord, mapDataSet


class MyDataSet(Dataset):
    """
    用于模型训练的数据集
    """

    def __init__(self, seqLen, imageClassDir, imageVectorDir, textDir, dataType=DATASET.TRAIN, isJoin=True):
        """
        :param seqLen:
        :param imageClassDir:图片对应类的字典序列化文件夹
        :param imageVectorDir:ResNet生成的文本向量的文件夹
        :param textDir:推特数据的文本数据文件夹
        :param isJoin:是否在concat 约为早期融合
        """
        self.isJoin = isJoin
        self.seqLen = seqLen
        self.imageVectorDir = imageVectorDir
        self.id2text = []
        #self.tokenizer = AutoTokenizer.from_pretrained(modelWightsDir + "bert-base-cased")
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        with open(textDir + "textVocab.py3", 'rb') as f:
            self.word2id = pickle.load(f)  # 词表
        with open(imageClassDir + "classVocab.py3", 'rb') as f:
            self.attribute2id = pickle.load(f)  # 类表
        with open(imageClassDir + "image2class.py3", 'rb') as f:
            self.dictExtractWords = pickle.load(f)  # 类表
        with open(textDir + mapDataSet[dataType], 'r', encoding="utf-8") as f:
            for line in f:
                if line.strip() == "":
                    continue
                if flitIllWord(line):  # 过滤非法word 20230513
                    self.id2text.append(eval(line))
        self.id2text = numpy.array(self.id2text)

    def processText(self, sqLen, source):
        """
        :param sqLen:文本长度
        :param source:字符串
        :return:对应词表的对应SqLen长度
        """
        strs = source.split(" ")
        if len(strs) > sqLen:
            strs = strs[:sqLen]
        strs = numpy.array(strs)
        func = numpy.vectorize(lambda x: self.word2id[x] if x in self.word2id else self.word2id['<unk>'])
        return numpy.pad(func(strs), (0, sqLen - len(strs)))

    def __getitem__(self, index):
        id = self.id2text[index][0]
        if self.isJoin:
            text = ' '.join(self.dictExtractWords[id]) + self.id2text[index][1]
        else:
            text = self.id2text[index][1]
        reText = torch.tensor(self.processText(self.seqLen, text), dtype=torch.int32)
        middle_Y = self.id2text[index][2]   ####改  numpy.str_ 不能转floatensor
        middle_y = numpy.array(middle_Y)    #### 先转numpy.array. 再用astype强制转换成float
        middle_y = middle_y.astype(float)
        retY = torch.tensor(middle_y, dtype=torch.float32)
        #retY = torch.tensor(self.id2text[index][2], dtype=torch.float32)
        reWords = torch.tensor(itemgetter(*self.dictExtractWords[id])(self.attribute2id))
        image = torch.tensor(numpy.load(self.imageVectorDir + id + ".npy").squeeze())  # numpy 矩阵
        # image = torch.load(self.imageVectorDir + id).squeeze()  # tensor 矩阵
        encodedInput = self.tokenizer(text, return_tensors='pt', padding="max_length", max_length=self.seqLen,
                                      truncation=True)  # 模型编码
        input_ids, token_type_ids, attention_mask = encodedInput["input_ids"], encodedInput[
            "token_type_ids"], encodedInput["attention_mask"]
        return (reText, image, reWords, (input_ids, token_type_ids, attention_mask)), retY, id

    def __len__(self):
        return self.id2text.shape[0]
