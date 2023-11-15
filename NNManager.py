import torch
from torch import nn

from codes.Attention import MultiModalAttention, DotProductAttention
from codes.ExtractFeature import ExtractFeature
from codes.Function import try_gpu
from codes.ImageFeature import ImageFeature
from codes.TextFeature import TextFeature


class Net(nn.Module):

    def __init__(self, nHidden, seqLen, dropout=0, numLayers=1, classEmbeddingPath="..//ExtractWords/vector",
                 textEmbeddingPath="../words/vector", device='cuda'):
        super().__init__()
        self.FinalMLPSize = 512
        self.image_size = 1024
        self.class_size = 200
        self.text_size = 512
        self.middle_size = 512
        self.device = device
        self.extractFeature = ExtractFeature(embeddingPath=classEmbeddingPath, device=device)  # 图像中物品类别
        self.imageFeature = ImageFeature()  # 图像特征
        self.imageFeature.apply(ImageFeature.weight_init)
        self.textFeature = TextFeature(nHidden, seqLen, textEmbeddingPath=textEmbeddingPath,
                                       numLayers=numLayers,
                                       guideLen=self.extractFeature.embSize, dropout=dropout, device=device)



        ######## 维度太大需要降维
        self.extractMatrix_fc = nn.Sequential(                      ## batchsize * 5 * 200
            nn.Flatten(),
            nn.Linear(self.class_size * 5, self.middle_size),
            nn.ReLU(inplace=True)                                   ## batchsize * middle_size
        )
        self.imageMatrix_fc = nn.Sequential(                         ##   batchsize * 196 * 1024
            nn.Flatten(),
            nn.Linear(self.image_size * 196, self.middle_size),
            # nn.ReLU(inplace=True),
            # nn.Linear(self.image_size * 14, self.image_size),
            # nn.Linear(self.image_size, self.middle_size),
            nn.ReLU(inplace=True)                                   ####### batchsize * self.middle_size
        )
        self.textMatrix_fc = nn.Sequential(                         ## batchsize * 80 * 512
            nn.Flatten(),
            nn.Linear(self.text_size * 80, self.text_size),
            # nn.ReLU(inplace=True),
            # nn.Linear(self.text_size*2, self.middle_size),
            nn.ReLU(inplace=True)                                   ## batchsize * middle_size
        )

        ####### 单个模态不融合 可视化conf_matrix


        ##### 一个全连接层 直接分类

        # self.conf_fc = nn.Sequential(
        #     nn.Linear(self.text_size, )
        # )


        # #### MMHIAN模块
        # 注意力机制以 x, y, z 指导向量计算与 key的评分，最后将其平均 这里用的是加性注意力机制，由seqToSeq翻译的注意力所启发
        # self.extractFeatureATT = MultiModalAttention(
        #     querySizes=(
        #         self.extractFeature.embSize, self.imageFeature.defaultFeatureSize, self.textFeature.nHidden * 2),
        #     keySize=self.extractFeature.embSize, dropout=dropout)
        # self.imageFeatureATT = MultiModalAttention(
        #     querySizes=(
        #         self.extractFeature.embSize, self.imageFeature.defaultFeatureSize, self.textFeature.nHidden * 2),
        #     keySize=self.imageFeature.defaultFeatureSize, dropout=dropout)
        # self.textFeatureATT = MultiModalAttention(
        #     querySizes=(
        #         self.extractFeature.embSize, self.imageFeature.defaultFeatureSize, self.textFeature.nHidden * 2),
        #     keySize=self.textFeature.nHidden * 2, dropout=dropout)

        
        ##### 每个单独融合
        # self.extractFeatureATT = nn.TransformerEncoderLayer(
        #     d_model=self.image_size + self.text_size + self.middle_size,
        #     nhead=8,
        #     dropout=0.1
        # )
        # self.imageFeatureATT = nn.TransformerEncoderLayer(
        #     d_model=self.class_size + self.text_size + self.middle_size,
        #     nhead=8,
        #     dropout=0.1
        # )
        # self.textFeatureATT = nn.TransformerEncoderLayer(
        #     d_model=self.class_size + self.image_size + self.middle_size,
        #     nhead=8,
        #     dropout=0.1
        # )
        ######### 融合extractGudieVec imageMatrix textMatrix        image_size1024换成middle_size512就能跑了
        self.img_txt_transformer = nn.TransformerEncoderLayer(
            d_model=self.class_size + self.middle_size + self.text_size,
            nhead=8,
            dropout=0.1
        )
        self.img_txt_transformer_fc = nn.Sequential(
            nn.Linear(self.class_size + self.middle_size + self.text_size, self.FinalMLPSize * 2),
            nn.ReLU(inplace=True)
        )

        # ######### 融合extractGuideVec,textGuideVec, imgGudieVec, textHmatrix, imageMatrix
        # self.transformer_attention = nn.TransformerEncoderLayer(
        #     d_model=self.class_size + self.image_size + self.text_size + self.middle_size * 2,
        #     nhead=8,
        #     dropout=0.1
        # )
        # self.transformer_fc = nn.Sequential(
        #     nn.Linear(self.class_size + self.text_size + self.image_size + self.middle_size * 2, self.FinalMLPSize * 3),
        #     nn.ReLU(inplace=True)
        # )


        # 为了后面的缩放点积注意力，需要把多模态向量调整为同一维度，后加入注意力机制，减少模型复杂度
        self.extractLinear = nn.Linear(self.middle_size + self.image_size + self.text_size, self.FinalMLPSize)
        self.extractRelu = nn.ReLU(inplace=True)
        self.imageLinear = nn.Linear(self.middle_size + self.text_size + self.class_size, self.FinalMLPSize)
        self.imageRelu = nn.ReLU(inplace=True)
        self.textLinear = nn.Linear(self.middle_size + self.image_size + self.class_size, self.FinalMLPSize)
        self.textRelu = nn.ReLU(inplace=True)


        ##### FET模块
        #self.multiAttention = DotProductAttention(dropout=dropout)
        ### 注意力机制可以使用transformer
        self.multiAttention = nn.TransformerEncoderLayer(d_model=self.FinalMLPSize * 3,
                                                         nhead=8,
                                                         dropout=0.1)
        #self.flatten = nn.Flatten()

        # 最后加入两层全连接层

        self.MLP, self.FC = nn.Linear(self.FinalMLPSize * 2, self.FinalMLPSize), nn.Linear(self.FinalMLPSize, 1)
        self.mlpRelu, self.fcSigmoid = nn.ReLU(), nn.Sigmoid()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)

    def forward(self, X):
        reText, images, reWords, text = X
        input_ids, token_type_ids, attention_mask = text

        if self.device == 'cuda':
            reText, images, reWords = reText.to(try_gpu()), images.to(try_gpu()), reWords.to(
                try_gpu())
            input_ids, token_type_ids, attention_mask = input_ids.to(try_gpu()), token_type_ids.to(
                try_gpu()), attention_mask.to(try_gpu())


        extractMatrix, extractGuidVec = self.extractFeature.forward(reWords)
        imageMatrix, imageGuidVec = self.imageFeature.forward(images)
        textHMatrix, textGuidVec = self.textFeature.forward(reText, (input_ids, token_type_ids, attention_mask),
                                                            extractGuidVec)
        extractGuidVec, imageGuidVec, textGuidVec = extractGuidVec.unsqueeze(1), imageGuidVec.unsqueeze(
            1), textGuidVec.unsqueeze(1)  # 升维

        extractMatrix = self.extractMatrix_fc(extractMatrix).unsqueeze(1)
        imageMatrix = self.imageMatrix_fc(imageMatrix).unsqueeze(1)
        textHMatrix = self.textMatrix_fc(textHMatrix).unsqueeze(1)

        # print(imageMatrix.shape)



# ***********************************************************************************************************************
        ######## HIAN模块 #############################
        # extractVec = self.extractFeatureATT.forward((extractGuidVec, imageGuidVec, textGuidVec), extractMatrix)
        # imageVec = self.imageFeatureATT.forward((extractGuidVec, imageGuidVec, textGuidVec), imageMatrix)
        # textVec = self.textFeatureATT.forward((extractGuidVec, imageGuidVec, textGuidVec), textHMatrix)  # QVA
        # extractVec = self.extractFeatureATT(torch.cat([imageGuidVec, textGuidVec, extractMatrix], dim=2))
        # imageVec = self.imageFeatureATT(torch.cat([textGuidVec, extractGuidVec, imageMatrix], dim=2))
        # textVec = self.textFeatureATT(torch.cat([imageGuidVec, extractGuidVec, textHMatrix], dim=2))
        ####################################################

        #
        # ####################### transformer encoder 注意力模块
        middle_Vec = self.img_txt_transformer(torch.cat([extractGuidVec, imageMatrix, textHMatrix], dim=2)).squeeze(1)
        # return middle_Vec
        final_Vec = self.img_txt_transformer_fc(middle_Vec)
        # return final_Vec
        # ###################################################
# ***************************************************************************************************************************


        ####################### 直接融合六部分    显存不够
        # middle_Vec = self.transformer_attention(torch.cat([extractGuidVec, imageGuidVec, textGuidVec, imageMatrix, textHMatrix], dim=2)).squeeze(1)
        # final_Vec = self.transformer_fc(middle_Vec)
        #############################################





        # extractVec, imageVec, textVec = extractVec.squeeze(1), imageVec.squeeze(1), textVec.squeeze(1)  # 降维

        # 是否加入relu继续激活 未实验 20230504
        # extractVec = self.extractLinear.forward(extractVec)
        # extractVec = self.extractRelu(extractVec)
        # imageVec = self.imageLinear.forward(imageVec)
        # imageVec = self.imageRelu(imageVec)
        # textVec = self.textLinear.forward(textVec)
        # textVec = self.textRelu(textVec)
        # #finalMatrix = torch.stack((extractVec, imageVec, textVec), dim=1)  # 转化为 batch * 3 * FinalMLPSize
        # finalMatrix = torch.cat([extractVec.unsqueeze(0), imageVec.unsqueeze(0), textVec.unsqueeze(0)], dim=2)
        # #finalVec = torch.mean(self.multiAttention.forward(finalMatrix), dim=1)
        # #finalVec = torch.mean(self.multiAttention(finalMatrix), dim=1)
        # finalVec = self.multiAttention(finalMatrix)
        # finalVec = finalVec.squeeze()



        # ###    baseline的t_nse         ################################
        # middle_Vec = torch.cat([extractGuidVec, imageMatrix, textHMatrix], dim=2).squeeze(1)
        # return middle_Vec             ##########################

##### 单模态直接分类 ****************************************************
        # ############ 图像模态
        # imageMatrix = imageMatrix.squeeze(1)
        # fcInput = self.mlpRelu(imageMatrix)
        # # return fcInput
        # return self.fcSigmoid(self.FC(fcInput))

        # ############# 文本模态
        # textMatrix = textHMatrix.squeeze(1)
        # fcInput = self.mlpRelu(textMatrix)
        # # return fcInput
        # return self.fcSigmoid(self.FC(fcInput))

        # # ############ 类别模态
        # extractMatrix = extractMatrix.squeeze(1)
        # fcInput = self.mlpRelu(extractMatrix)
        # return self.fcSigmoid(self.FC(fcInput))

        # #### 三者cat
        # fcinput = torch.cat([extractGuidVec.squeeze(1), imageMatrix.squeeze(1), textHMatrix.squeeze(1)], dim=1)
        # fcInput = self.MLP(self.mlpRelu(self.img_txt_transformer_fc(fcinput)))
        # return self.fcSigmoid(self.FC(fcInput))
######### *****************************************************







        #
        fcInput = self.mlpRelu(self.MLP(final_Vec))
        # return fcInput
        return self.fcSigmoid(self.FC(fcInput))
