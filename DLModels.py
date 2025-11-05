import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class BertEmbedding(nn.Module): # BERT 嵌入模块 继承 nn.Module
    def __init__(self, bert_model):
        """
        BERT 嵌入模块
        Args:
            bert_model (BertModel): BERT 模型
        """
        super(BertEmbedding, self).__init__() # 继承 nn.Module 的初始化方法
        self.bert_model = bert_model

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids (Tensor): 输入序列 (batch_size, sequence_length)
            attention_mask (Tensor): 注意力掩码 (batch_size, sequence_length)

        Returns:
            sentence_embeddings (Tensor): 句子嵌入 (batch_size, num_sentences, embedding_dim)
        """
        # 调用 BERT 模型获取句子嵌入
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embeddings = outputs[0]  # (batch_size, sequence_length, embedding_dim)
        return sentence_embeddings

class Attention(nn.Module): # Attention 模块 继承 nn.Module
    def __init__(self, hidden_dim, attn_method='dot_scaled', scale_factor=None):
        """
        Attention 模块
        Args:
            hidden_dim (int): GRU 隐层大小 
        """
        super(Attention, self).__init__() # 继承 nn.Module 的初始化方法
        self.attn_method = attn_method
        # 定义注意力权重
        if self.attn_method == 'dot':
            self.attention_weight = nn.Linear(2 * hidden_dim, 1, bias=False)
        elif self.attn_method == 'dot_scaled':
            self.scale_factor = scale_factor if scale_factor is not None else torch.sqrt(torch.tensor(2 * hidden_dim, dtype=torch.float32))
            self.attention_weight = nn.Linear(2 * hidden_dim, 1, bias=False)
        elif self.attn_method == 'additative':
            self.attention_weight = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
            self.v = nn.Parameter(torch.randn(hidden_dim, 1))
        else:
            raise ValueError(f"Unsupported attention method: {self.attn_method}")
        
    def forward(self, gru_output):
        """
        Args:
            gru_output (Tensor): GRU 输出 (batch_size, sequence_length, 2 * hidden_dim)

        Returns:
            context_vector (Tensor): 加权后的表示 (batch_size, 2 * hidden_dim)
            attention_weights (Tensor): 注意力权重 (batch_size, sequence_length, 1)
        """
        if self.attn_method == 'dot':
            weights = self.attention_weight(gru_output)  # (batch_size, sequence_length, 1)
        elif self.attn_method == 'dot_scaled':
            weights = self.attention_weight(gru_output) / self.scale_factor  # (batch_size, sequence_length, 1)
        elif self.attn_method == 'additative':
            # 首先通过一个非线性变换
            hidden = torch.tanh(self.attention_weight(gru_output))  # (batch_size, sequence_length, hidden_dim)
            # 然后计算注意力权重
            weights = torch.matmul(hidden, self.v)  # (batch_size, sequence_length, 1)
        
        weights = torch.softmax(weights, dim=1)  # (batch_size, sequence_length, 1)
        context_vector = torch.sum(weights * gru_output, dim=1)  # 加权求和 (batch_size, 2 * hidden_dim)
        return context_vector, weights


class GBN(nn.Module): #缩减后的模型，继承nn.Module，输入CLS向量，只保留句子级GRU

    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout=0.5,num_classes=2,attn_method='dot_scaled'): # 输入参数
        """
        GBN 模型

        Args:
            embedding_dim (int): 输入嵌入的维度
            hidden_dim (int): GRU 隐层大小
            num_layers (int): GRU 的层数
            dropout (float): Dropout 比例
        """
        super(GBN, self).__init__() # 继承 nn.Module 的初始化方法
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 文档级 GRU 和 Attention
        self.document_gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.document_attention = Attention(hidden_dim,attn_method='dot_scaled')
        self.document_dropout = nn.Dropout(dropout)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 使用自适应平均池化层
        # 添加一个线性层，用于分类输出
        self.fc = nn.Linear(2 * hidden_dim, num_classes)
        
        # 添加 softmax 层
        self.log_softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size,max_sentences):
        """初始化隐藏状态 at t=0

        Args:
            batch_size: int, the size of the current evaluated batch
        """

        h0 = torch.zeros(batch_size, max_sentences, 2*self.hidden_dim).to(device=self.device)
        
        return(h0)

    def forward(self, cls_embeddings):
        """
        Args:
            cls_embeddings (Tensor): 输入嵌入 (batch_size, num_sentences, embedding_dim)

        Returns:
            document_representation (Tensor): 文档表示 (batch_size, 2 * hidden_dim)
            attention_weights (list): 注意力权重 [sentence_weights, document_weights]
        """
        
        # sentence_embeddings 形状: (batch_size, num_sentences, num_words, 768)
        batch_size, num_sentences, _ = cls_embeddings.shape

        # 文档级编码
        document_output, _ = self.document_gru(cls_embeddings)
        document_output = self.document_dropout(document_output) #输出形状为(batch_size, num_sentences, 2*hidden_dim)
        document_weights = None #HEm模型没有句子级Attention

        # 应用平均池化层
        document_representation = self.avg_pool(document_output.transpose(1, 2)).squeeze(2)  # 形状: (batch_size, 2 * hidden_dim)
        # transpose是因为avg_pool在最后一个维度上进行池化，所以我们需要先转换维度

        # 分类输出
        logits = self.fc(document_representation)  # (batch_size, num_classes)

        # 应用 softmax 层
        probabilities = self.log_softmax(logits)

        return probabilities, [document_weights]

class GBAN(nn.Module): # GBAN 模型 继承 nn.Module
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout=0.5,num_classes=2,attn_method='dot_scaled'): # 输入参数
        """
        GBAN 模型

        Args:
            embedding_dim (int): 输入嵌入的维度
            hidden_dim (int): GRU 隐层大小
            num_layers (int): GRU 的层数
            dropout (float): Dropout 比例
        """
        super(GBAN, self).__init__() # 继承 nn.Module 的初始化方法
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 文档级 GRU 和 Attention
        self.document_gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.document_attention = Attention(hidden_dim,attn_method=attn_method)
        self.document_dropout = nn.Dropout(dropout)

        # 添加一个线性层，用于分类输出
        self.fc = nn.Linear(2 * hidden_dim, num_classes)
        
        # 添加 softmax 层
        self.log_softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size,max_sentences):
        """初始化隐藏状态 at t=0

        Args:
            batch_size: int, the size of the current evaluated batch
        """

        h0 = torch.zeros(batch_size, max_sentences, 2*self.hidden_dim).to(device=self.device)
        
        return(h0)

    def forward(self, cls_embeddings):
        """
        Args:
            cls_embeddings (Tensor): 输入嵌入 (batch_size, num_sentences, embedding_dim)

        Returns:
            document_representation (Tensor): 文档表示 (batch_size, 2 * hidden_dim)
            attention_weights (list): 注意力权重 [sentence_weights, document_weights]
        """
        
        # sentence_embeddings 形状: (batch_size, num_sentences, num_words, 768)
        batch_size, num_sentences, _ = cls_embeddings.shape

        # 文档级编码
        document_output, _ = self.document_gru(cls_embeddings)
        document_output = self.document_dropout(document_output)

        # 文档级 Attention
        document_representation, document_weights = self.document_attention(document_output)
        # 分类输出
        logits = self.fc(document_representation)  # (batch_size, num_classes)

        # 应用 softmax 层
        probabilities = self.log_softmax(logits)

        return probabilities, [document_weights]

class GBAN_5_categories(nn.Module): # GBAN 模型 继承 nn.Module
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout=0.5,num_classes=5,attn_method='dot_scaled'): # 输入参数
        """
        GBAN 模型 5分类

        Args:
            embedding_dim (int): 输入嵌入的维度
            hidden_dim (int): GRU 隐层大小
            num_layers (int): GRU 的层数
            dropout (float): Dropout 比例
        """
        super(GBAN_5_categories, self).__init__() # 继承 nn.Module 的初始化方法
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 文档级 GRU 和 Attention
        self.document_gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.document_attention = Attention(hidden_dim,attn_method=attn_method)
        self.document_dropout = nn.Dropout(dropout)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 使用自适应平均池化层
        # 添加一个线性层，用于分类输出
        self.fc = nn.Linear(2 * hidden_dim, num_classes)
        
        # 添加 softmax 层
        self.log_softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size,max_sentences):
        """初始化隐藏状态 at t=0

        Args:
            batch_size: int, the size of the current evaluated batch
        """

        h0 = torch.zeros(batch_size, max_sentences, 2*self.hidden_dim).to(device=self.device)
        
        return(h0)

    def forward(self, cls_embeddings):
        """
        Args:
            cls_embeddings (Tensor): 输入嵌入 (batch_size, num_sentences, embedding_dim)

        Returns:
            document_representation (Tensor): 文档表示 (batch_size, 2 * hidden_dim)
            attention_weights (list): 注意力权重 [sentence_weights, document_weights]
        """
        
        # sentence_embeddings 形状: (batch_size, num_sentences, num_words, 768)
        batch_size, num_sentences, _ = cls_embeddings.shape

        # 文档级编码
        document_output, _ = self.document_gru(cls_embeddings)
        document_output = self.document_dropout(document_output)

        # 文档级 Attention
        document_representation, document_weights = self.document_attention(document_output)
        # 分类输出
        logits = self.fc(document_representation)  # (batch_size, num_classes)

        # 应用 softmax 层
        probabilities = self.log_softmax(logits)

        return probabilities, [document_weights]

class GBN_5_categories(nn.Module): # HEA 模型 继承 nn.Module
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout=0.5,num_classes=5,attn_method='dot_scaled'): # 输入参数
        """
        GBN模型 5分类

        Args:
            embedding_dim (int): 输入嵌入的维度
            hidden_dim (int): GRU 隐层大小
            num_layers (int): GRU 的层数
            dropout (float): Dropout 比例
        """
        super(GBN_5_categories, self).__init__() # 继承 nn.Module 的初始化方法
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 文档级 GRU 和 Attention
        self.document_gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.document_attention = Attention(hidden_dim,attn_method=attn_method)
        self.document_dropout = nn.Dropout(dropout)

        # 添加一个线性层，用于分类输出
        self.fc = nn.Linear(2 * hidden_dim, num_classes)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 使用自适应平均池化层
        # 添加 softmax 层
        self.log_softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size,max_sentences):
        """初始化隐藏状态 at t=0

        Args:
            batch_size: int, the size of the current evaluated batch
        """

        h0 = torch.zeros(batch_size, max_sentences, 2*self.hidden_dim).to(device=self.device)
        
        return(h0)

    def forward(self, cls_embeddings):
        """
        Args:
            cls_embeddings (Tensor): 输入嵌入 (batch_size, num_sentences, embedding_dim)

        Returns:
            document_representation (Tensor): 文档表示 (batch_size, 2 * hidden_dim)
            attention_weights (list): 注意力权重 [sentence_weights, document_weights]
        """
        
                # sentence_embeddings 形状: (batch_size, num_sentences, num_words, 768)
        batch_size, num_sentences, _ = cls_embeddings.shape

        # 文档级编码
        document_output, _ = self.document_gru(cls_embeddings)
        document_output = self.document_dropout(document_output) #输出形状为(batch_size, num_sentences, 2*hidden_dim)
        document_weights = None #HEm模型没有句子级Attention

        # 应用平均池化层
        document_representation = self.avg_pool(document_output.transpose(1, 2)).squeeze(2)  # 形状: (batch_size, 2 * hidden_dim)
        # transpose是因为avg_pool在最后一个维度上进行池化，所以我们需要先转换维度

        # 分类输出
        logits = self.fc(document_representation)  # (batch_size, num_classes)
        # 应用 softmax 层
        probabilities = self.log_softmax(logits)

        return probabilities, [document_weights]

if __name__ == '__main__':
    embedding_dim = 768
    hidden_dim = 128
    num_layers = 2
    dropout = 0.5

    # 初始化模型
    # 测试输出的维度是否满足要求

    GBN_model = GBN(embedding_dim, hidden_dim, num_layers, dropout)
    GBAN_model = GBAN(embedding_dim, hidden_dim, num_layers, dropout)

    # 假设输入
    sentence_embeddings = torch.randn(8, 20, 512,768)  # (batch_size=8, num_sentences=20, num_words=512, embedding_dim=768)
    sentence_embeddings_cls = torch.randn(8, 20, 768)  


    # GBN 模型前向传播
    document_representation, attention_weights = GBN_model(sentence_embeddings_cls)
    print("GBN Document Representation:", document_representation.shape)

    #GBAN 模型前向传播
    document_representation, attention_weights = GBAN_model(sentence_embeddings_cls)
    print("GBAN Document Representation:", document_representation.shape)
    print("Attention Weights:", len(attention_weights))
