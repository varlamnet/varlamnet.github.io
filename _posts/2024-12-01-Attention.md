---
layout: post
title: Attention
subtitle: For Transformers
thumbnail-img: assets/img/soft.jpg
comments: false
---

<!-- MathJAx Import -->

$$\newcommand{\abs}[1]{\left\lvert#1\right\rvert}$$
$$\newcommand{\norm}[1]{\left\lVert#1\right\rVert}$$
$$\newcommand{\inner}[1]{\left\langle#1\right\rangle}$$
$$\DeclareMathOperator*{\argmin}{arg\,min}$$
$$\DeclareMathOperator*{\argmax}{arg\,max}$$
$$\DeclareMathOperator*{\E}{\mathbb{E}}$$
$$\DeclareMathOperator*{\V}{\mathbb{V}}$$
$$\DeclareMathOperator*{\x}{\mathbb{x}}$$

<!-- MathJAx End -->
<p style="margin-bottom:-2cm;"></p>

<p class="dropcap">A</p> ttention was first mentioned around 2014 (e.g., [additive attention paper](https://arxiv.org/pdf/1409.0473) and [scaled attention paper](https://arxiv.org/pdf/1508.04025)).
The specific type of attention implemented in ["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762) paper has become known as Scaled Dot Product Attention.

Attention is a mechanism that allows to determine related parts in the input sequence (e.g., sentence) so that the model could "attend" to (pay attention to) related pieces of the sequence. It is the key component behind the success of transformers in tasks like NLP, CV, etc.

{: .box-note}
Notation:
<span style="display:block; height: 10px;"></span>
$n: = n_{seq}$ -- sequence length (elements in input sequence).
<span style="display:block; height: 10px;"></span>
$d_{model}$ -- embedding dimension.
<span style="display:block; height: 10px;"></span>
$h := n_{heads}$ -- number of heads.
<span style="display:block; height: 10px;"></span>
$d_k = d_q := d_{model}/n_{heads}$ -- key (query) matrix dimension.
<span style="display:block; height: 10px;"></span>
$d_v := d_{model}/n_{heads}$ -- value matrix dimension.
<span style="display:block; height: 10px;"></span>

## Single-Head Attention

_Single-head_ scaled dot product attention is calculated over the entire $d_{model}$ embedding space (without partitioning it).
Formally, attention is defined as,

$$Attention(Q, K, V) = \text{softmax}(\frac{Q K^T}{\sqrt{d_k}})V, $$

where $Q \in\mathbb{R}^{n\times d_{model}}$, $K \in\mathbb{R}^{n\times d_{model}}$, $V \in\mathbb{R}^{n\times d_{model}}$.

- $Q$ represent the user query or question and is used to determine the relevance of other words.
- $K$ stores the answers to other queries and is used to measure similarity of the query to other queries.
- $V$ are the actual answers stored and are weighted based on similarity of $Q$ and $K$ to generate the final output.

Matrix $\frac{Q K^T}{\sqrt{d_k}}$ is known as **attention scores**. These represent (scaled) dot products and hence measure how close the words are in the embedding space. The division by $\sqrt{d_k}$ is to counterbalance the growing values of the dot product when $d_k$ dimension is large and thus stabilize the gradient.

When softmax is applied, $\text{softmax}(\frac{Q K^T}{\sqrt{d_k}})$ provides **attention weights**. This is simply an $n\times n$ matrix, where $n$ is the sequence length (number of words), and is often visualized to gain insights about how the model thinks. Softmax ensures that each row sums to 1 and hence these values can be seen as weights. The entry $a_{ij}$ will characterized the "share" of similarity between words $i$ and $j$, with $\sum_j a_{ij} = 1$. The attention weight matrix is what fills each word in a sentence with contextual information.

Notice that, in case of **self-attention**, i.e., when all three inputs are equal $Q=K=V$, the output of attention is simply a weighted average of the input, where the weight is determined by the similarity of the input with itself. In other words, $i$th self-attention of the input vector $\x$ is simply $a_i = \sum_j f(\x_i^T \x_j) \x_j$, where $f$ is the softmax function. Put this way, this hints to similarities with the kernel regression, which is a topic for another discussion.

## Intuition

In a sentence, _"I bought a baseball bat"_, attention mechanism helps the model realize that _"bat"_ refers to a wooden object rather than an animal. This happens through the adjustment of the embedding vector for _"bat"_ since it will interact (through dot product) with the word _"baseball"_ .

Let's focus on words _"baseball"_ and _"bat"_ only. For the sake of this example, consider a 3-dimensional embedding space, where the 3 dimensions are responsible for "sports", "animal" and "other". The word _"baseball"_ would have an embedding vector of something like $(2, 0, 0)$, while _"bat"_ would have $(1, 1, 0)$, i.e., by itself alone it is difficult to tell whether _"bat"_ refers to an object (sports) or an animal. The attention will help disambiguate that.

For simplicity, let's consider self-attention, in which we pass $(n, d_{model})$ input $Q=K=V$ cloned 3 times. Self-attention is computed in both encoder and decoder parts of transformer architecture. The initial key matrix with $n = 2$ and $d_k = d_q = d_{model}= 3$ is given as,

<center>
<a style="padding: 1rem;">Query :</a>  
$\begin{matrix}  \textit{baseball} & [2 & 0 & 0]\\ \textit{bat} & [1 & 1 & 0] \end{matrix}$
<span style="padding: 1rem;"></span>
</center>

Then, we can do a (scaled) matrix multiplication to get attention scores and apply softmax (row-wise) to get attention weights,

<center>
<a style="padding: 1rem;">Scores:</a>  
$\begin{matrix}  \textit{baseball} & [2.31 & 1.15]\\ \textit{bat} & [1.15 & 1.15] \end{matrix}$  
<a style="padding: 2rem;">Weights:</a>  
$\begin{matrix}  \textit{baseball} & [.76 & .24]\\ \textit{bat} & [.50 & .50] \end{matrix}$  
<span style="padding: 1rem;"></span>
</center>

Notice that the attention scores and weights are $n \times n$ (sequence length) dimensional -- these matrices are often examined to make the model more interpretable. The weight for _"bat"_ indicates that its embedding will be pulled towards _"baseball"_ (which has a large "sports" dimension and small "animal" direction). Finally, self-attention is obtained by multiplying attention weights with values,

$$\begin{matrix}  \textit{baseball} & [1.76 & .24 & 0]\\ \textit{bat} & [1.50 & .50 & 0] \end{matrix}$$

Hence, the embedding vector for bat is now "pulled" towards baseball and the model can infer that _"bat"_ probably refers to a wooden object rather than an animal. This is how attention injects the contextual information.

Or in code (with deepseek's help),

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        """
        Scaled Dot-Product Attention proposed in "Attention Is All You Need"
        """
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask=None, dropout=None):
        """
        Forward pass for scaled dot-product attention.

        Args:
            Q (torch.Tensor): Query shape (batch, n_heads, seq_len, d_k).
            K (torch.Tensor): Key shape (batch, n_heads, seq_len, d_k).
            V (torch.Tensor): Value shape (batch, n_heads, seq_len, d_k).
            mask (torch.Tensor, optional): Mask padding or future tokens.
            dropout (nn.Dropout, optional): Dropout layer.

        Returns:
            torch.Tensor: Output shape (batch, n_heads, seq_len, d_k).
            torch.Tensor: Att'n weights shape (batch, n_heads, seq_len, seq_len).
        """
        d_k = Q.shape[-1]
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        if dropout is not None:
            weights = dropout(weights)
        return (weights @ V), weights


query = torch.Tensor([[2, 0, 0], [1, 1, 0]])
scaled = ScaledDotProductAttention()
scaled.forward(Q=query, K=query, V=query)

# (tensor([[1.7604, 0.2396, 0.0000],
#          [1.5000, 0.5000, 0.0000]]),
#  tensor([[0.7604, 0.2396],
#          [0.5000, 0.5000]]))
```

{: .box-warning}
Note,
<span style="display:block; height: 10px;"></span>
⦁ Pytorch provides an [implementation of MultiHeadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
<span style="display:block; height: 10px;"></span>
⦁ The masking layer is required to remove parts of attention, mainly to prevent the model from leaking its predictions into the future when generating output.
<span style="display:block; height: 10px;"></span>
⦁ Dropout layer is optional but often added in practice.

**Self-attention** is used in both encoder and decoder of the transformer. In encoder it can look both backwards and forward in the sequence to determine the importance. In decoder it can only look backwards to avoid leaking information when generating output.
In addition, there is also **cross-attention** in decoder -- it takes the decoder's query but key & values come from encoder.

## Multi-Head Attention

Compared to single-head attention, multi-head attention adds 2 important features,

- **Multiple Heads:** multi-head attention breaks $Q$, $K$ and $V$ across the embedding dimension into $d_{model}/h$ chunks ($d_{model}=512$ and $h=8$ in the original paper). This helps gather different types of context across different parts of the embedding space. This also permits attention to be computed in parallel across $h$ heads since the heads are not dependent on each other.

- **Projection Parameters:** the chunks are then projected into learnable matrices $W^Q$, $W^K$ and $W^V$. Projections add flexibility since it can be easier to gather context in projected space rather than the original space. Attention is applied to these projected chunks, everything is concatenated back into a single piece and projected again onto another learnable weight matrix $W^O$. In practice, these weights are usually Xavier initiated.

More formally,

$$MultiHead(Q, K, V) = Concat(head_1, \ldots, head_{h}) \, W^O,$$

$$head_i = Attention(Q W^Q_i, \, K W^K_i, \, V W^V_i)$$

where $Q \in\mathbb{R}^{n\times d_{q}}$, $K \in\mathbb{R}^{n\times d_{k}}$, $V \in\mathbb{R}^{n\times d_{v}}$ and $d_k = d_q  = d_v = d_{model}/h$.
Learnable projection weights are $W^Q_i \in\mathbb{R}^{d_{model}\times d_q}$, $W^K_i \in\mathbb{R}^{d_{model}\times d_k}$, $W^V_i \in\mathbb{R}^{d_{model}\times d_v}$, and $W^O_i \in\mathbb{R}^{d_{model}\times d_{model}}$.

Code implementation below.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        """
        Multi-Head Attention proposed in "Attention Is All You Need".

        Args:
            d_model (int): Dimensionality of the model embeddings.
            n_heads (int): Number of attention heads.
            dropout (float): Dropout share.
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        assert self.d_k * n_heads == d_model, "d_model must be divisible by n_heads"

        # Linear layers to project input embeddings
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attention = ScaledDotProductAttention()

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for multi-head attention.

        Args:
            query (torch.Tensor): Shape (batch, seq_len, d_model).
            key (torch.Tensor): Shape (batch, seq_len, d_model).
            value (torch.Tensor): Shape (batch, seq_len, d_model).
            mask (torch.Tensor, optional): For masking future tokens.

        Returns:
            torch.Tensor: Output of shape (batch, seq_len, d_model).
        """
        batch = query.shape[0]

        # Linear transformations to project inputs
        Q = self.W_q(query)  # -> (batch, seq_len, d_model)
        K = self.W_k(key)  # -> (batch, seq_len, d_model)
        V = self.W_v(value)  # -> (batch, seq_len, d_model)

        # Reshape Q, K, V to split into multiple heads
        # -> (batch, n_heads, seq_len, d_k)
        Q = Q.view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Apply scaled dot-product attention
        out, self.attention_weights = self.attention(Q, K, V, mask, self.dropout)

        # Reshape and concatenate outputs from all heads
        # -> (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch, -1, self.d_model)

        return self.W_o(out)
```

<span style="display:block; height: 0px;"></span>

## ~~Not So~~ Recent developments

Notice that self-attention computation time is $O(n^2)$ with respect to input sequence length $n$. As transformers have started getting larger, there is been some effort to try to improve that.

- Sparse attention -- e.g., see [this paper on sparse factorization of Attention matrix](https://arxiv.org/pdf/1904.10509).
- Sliding window attention -- e.g., [Longformer paper](https://arxiv.org/pdf/2004.05150) that attempts to reduce attention matrix computation to $O(n)$.
- [Grouped-query attention](https://arxiv.org/pdf/2305.13245) builds on multi-query attention and accelerates attention computation by sharing key-value matrices between all heads.

<span style="display:block; height: 0px;"></span>

## Why attention is all you need?

Attention offers the following benefits,

- **Long-Range Dependencies:** Attention enables the model to identify and utilize relationships between tokens, no matter how far apart they are in the sequence.
- **Parallelization:** Unlike sequential models like RNNs, attention computations can be performed in parallel across all elements of the sequence.
- **Interpretability:** The attention weights provide a clear view of which parts of the input the model is focusing on. In addition, attention can be adapted to various tasks and data types, making it a versatile tool in AI.
