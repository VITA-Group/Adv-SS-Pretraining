''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import attention_pooling.Constants as Constants
from attention_pooling.Layers import EncoderLayer, DecoderLayer

__author__ = "Yu-Hsiang Huang"

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)
import pdb

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


#non_ = get_non_pad_mask(src)
#print(non_)
#atten_ = get_attn_key_pad_mask(src, src)
#print(atten_)
def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

class attention_pooling(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

#        n_position = len_max_seq + 1

#        self.src_word_emb = nn.Embedding(
#            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

#        self.position_enc = nn.Embedding.from_pretrained(
#            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
#            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, return_attns=False):

        enc_slf_attn_list = []
#        pdb.set_trace()
#        print(src_seq)
        # -- Prepare masks
#        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
#        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = src_seq
        
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=1,
                slf_attn_mask=None)
#            print(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output[:, 0, :]


class SelfieModel(nn.Module):
    def __init__(
            self,
            n_layers, n_heads, d_in, d_model,
            d_ff, n_split, dropout=0.1, use_cuda=True, gpu = None):
        super(SelfieModel, self).__init__()
        self.n_split = n_split
        self.at_pool = at_pool = attention_pooling(n_layers + 1, n_heads, d_in, d_in, d_model, d_ff)

        if use_cuda:
            if gpu is None:
                self.row_embeddings = nn.Parameter(torch.randn(n_split, d_model).cuda())
                self.column_embeddings = nn.Parameter(torch.zeros(n_split, d_model).cuda())
                self.u0 = nn.Parameter(torch.zeros(1,1,d_model).cuda())
            else:
                self.row_embeddings = nn.Parameter(torch.randn(n_split, d_model).cuda(gpu))
                self.column_embeddings = nn.Parameter(torch.zeros(n_split, d_model).cuda(gpu))
                self.u0 = nn.Parameter(torch.zeros(1,1,d_model).cuda(gpu))
        else:
            self.row_embeddings = nn.Parameter(torch.randn(n_split, d_model))
            self.column_embeddings = nn.Parameter(torch.zeros(n_split, d_model))
            self.u0 = nn.Parameter(torch.zeros(1,1,d_model))

    def forward(self, src_seq, pos, return_attns=False):
        u = self.u0.repeat((src_seq.shape[0], 1, 1))
        src_seq = torch.cat([u, src_seq], dim=1)
        before_embeddings =  self.at_pool(src_seq)
        final = []
        rows = map(lambda x: np.trunc(x / self.n_split).astype("int"), pos)
        cols = map(lambda x: np.mod(x, self.n_split).astype("int"), pos)
        for (i,j) in zip(rows, cols):
            sum_up = before_embeddings + self.row_embeddings[i,:] + self.column_embeddings[j,:]
            sum_up = sum_up.unsqueeze(1)
            final.append(sum_up)

        return torch.cat(final, 1)
#pdb.set_trace()
#print(Constants.PAD)
if __name__ == '__main__':
    src = torch.ones((512, 12, 1024))#batch_size, encode_patch_size, vector length after P
    n_layers = 12
    d_model = 1024#vector length after the patch routed in P
    d_in = 64
    n_heads = d_model// d_in
    d_ff = 1024
    at_pool = SelfieModel(n_layers, n_heads, d_in, d_in, d_model, d_ff, 4, use_cuda=False)
    out = at_pool(src, [3,4,8])
    print(out.shape)
















