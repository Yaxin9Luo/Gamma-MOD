def flops_llama_layer(seq_len,hidden_size=4096,intermediate_size=11008):
    flops_attn=8*seq_len*hidden_size*hidden_size+4*seq_len*seq_len*hidden_size
    flops_ffn=3*2*seq_len*hidden_size*intermediate_size
    return flops_attn+flops_ffn


# vocab_size=32000
N_DENSE=3
N_MOD=29
R_MOD=0.379
SEQ_LEN=1400 #用数据集大概的一个平均值

flops_per_dense_layer=flops_llama_layer(SEQ_LEN)

flops_per_mod_layer=flops_llama_layer(int(SEQ_LEN*(1-R_MOD)))

total_flops=N_DENSE*flops_per_dense_layer+N_MOD*flops_per_mod_layer

dense_flops=(N_DENSE+N_MOD)*flops_per_dense_layer

print('dense flops: %.2f BFlops'%(dense_flops/1e12))

print('current flops: %.2f BFlops'%(total_flops/1e12))

