import torch
from tqdm import tqdm
import numpy as np
# config: num_layers, d_model, d_probability, layer_norm_ep, dropout_rate, max_position_embeddings, d_ff, d_kv, num_heads

# config 1
class MyConfig():
    def __init__(self):
        self.num_layers = 16
        self.d_model = 256
        self.d_probability = 128
        self.layer_norm_ep = 1e-5
        self.dropout_rate = 0.02
        self.max_position_embeddings = 30000
        self.d_ff = 512
        self.d_kv = 256
        self.num_heads = 8

# # config 2
# class MyConfig():
#     def __init__(self):
#         self.num_layers = 4
#         self.d_model = 256
#         self.d_probability = 256
#         self.layer_norm_ep = 1e-5
#         self.dropout_rate = 0.02
#         self.max_position_embeddings = 30000
#         self.d_ff = 512
#         self.d_kv = 48
#         self.num_heads = 4

# class MyConfig():
#     def __init__(self):
#         self.num_layers = 1
#         self.d_model = 64
#         self.d_probability = 64
#         self.layer_norm_ep = 1e-5
#         self.dropout_rate = 0.02
#         self.max_position_embeddings = 30000
#         self.d_ff = 128
#         self.d_kv = 48
#         self.num_heads = 2


class Attention(torch.nn.Module):
    def __init__(self, config, is_decoder):
        super(Attention, self).__init__()
        self.is_decoder = is_decoder
        self.d_model = config.d_model
        self.k_v_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.k_v_proj_dim
        self.q = torch.nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = torch.nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = torch.nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = torch.nn.Linear(self.inner_dim, self.d_model, bias=False)
        self.dropout1 = torch.nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, key_value_states=None):
        batch_size, seq_len = hidden_states.shape[:2]
        real_seq_len = seq_len
        key_len = real_seq_len if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.k_v_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            else:
                hidden_states = shape(proj_layer(key_value_states))
            return hidden_states

        # get query states
        if not self.is_decoder:
            query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        else:
            query_states = shape(self.q(key_value_states))

            # get key/value states
        key_states = project(hidden_states, self.k, key_value_states)
        value_states = project(hidden_states, self.v, key_value_states)

        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        # if self.is_decoder:
        #     scores = torch.tril(scores, diagonal=0)
        scores = torch.tril(scores, diagonal=-999999)
        attn_weights = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = self.dropout1(attn_weights)
        attn_output = unshape(torch.matmul(attn_weights, value_states))
        attn_output = self.o(attn_output)
        present_key_value_state = (key_states, value_states) if self.is_decoder else None
        outputs = (attn_output,) + (present_key_value_state, )
        return outputs


class SelfAttentionLayer(torch.nn.Module):
    def __init__(self, config):
        super(SelfAttentionLayer, self).__init__()
        self.SelfAttention = Attention(config, is_decoder=False)
        self.layer_norm = torch.nn.LayerNorm(config.d_model, eps=config.layer_norm_ep)
        self.dropout = torch.nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(hidden_states=normed_hidden_states)
        hidden_states = hidden_states + self.dropout(attention_output[0])
        return hidden_states


class CrossAttentionLayer(torch.nn.Module):
    def __init__(self, config):
        super(CrossAttentionLayer, self).__init__()
        self.CrossAttention = Attention(config, is_decoder=True)
        self.layer_norm = torch.nn.LayerNorm(config.d_model, eps=config.layer_norm_ep)
        self.dropout = torch.nn.Dropout(config.dropout_rate)

    def forward(self, encoder_hidden_states, decoder_hidden_states):
        normed_encoder_hidden_states = self.layer_norm(encoder_hidden_states)
        attention_output = self.CrossAttention(hidden_states=normed_encoder_hidden_states,
                                               key_value_states=decoder_hidden_states)
        outputs = encoder_hidden_states + self.dropout(attention_output[0])
        return outputs


class DenseReluDense(torch.nn.Module):
    def __init__(self, config):
        super(DenseReluDense, self).__init__()
        self.wi = torch.nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = torch.nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = torch.nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = torch.nn.functional.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class FFLayer(torch.nn.Module):
    def __init__(self, config):
        super(FFLayer, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(config.d_model, eps=config.layer_norm_ep)
        self.DenseReluDense = DenseReluDense(config)
        self.dropout = torch.nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forward_states = self.layer_norm(hidden_states)
        forward_states = self.DenseReluDense(forward_states)
        hidden_states = hidden_states + self.dropout(forward_states)
        return hidden_states


class AttentionBlock(torch.nn.Module):
    def __init__(self, config, is_decoder):
        super(AttentionBlock, self).__init__()
        self.is_decoder = is_decoder
        self.layer = torch.nn.ModuleList()
        self.layer.append(SelfAttentionLayer(config))
        if is_decoder:
            self.layer.append(CrossAttentionLayer(config))
        self.layer.append(FFLayer(config))

    def forward(self, encoder_hidden_states, decoder_hidden_states=None):
        outputs = None
        if not self.is_decoder:
            self_attention_vectors = self.layer[0](encoder_hidden_states)
            outputs = self.layer[1](self_attention_vectors)
        else:
            self_attention_vectors = self.layer[0](decoder_hidden_states)
            cross_attention_vectors = self.layer[1](encoder_hidden_states=encoder_hidden_states,
                                                    decoder_hidden_states=self_attention_vectors)
            outputs = self.layer[2](cross_attention_vectors)
        return outputs


class PositionEmbeddings(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = torch.nn.Embedding(config.max_position_embeddings, config.d_model)
        self.dropout = torch.nn.Dropout(config.dropout_rate)

    def forward(self, position_ids):
        position_embeddings = self.embedding(position_ids)
        position_embeddings = self.dropout(position_embeddings)
        return position_embeddings


class AttentionStack(torch.nn.Module):
    def __init__(self, config, is_decoder):
        super(AttentionStack, self).__init__()
        self.block = torch.nn.ModuleList([AttentionBlock(config, is_decoder) for i in range(config.num_layers)])
        self.final_layer_norm = torch.nn.LayerNorm(config.d_model, eps=config.layer_norm_ep)
        self.dropout = torch.nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, decoder_hidden_states=None):
        for i, layer_module in enumerate(self.block):
            hidden_states = layer_module(hidden_states,
                                         decoder_hidden_states=decoder_hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
    

class MyModel(torch.nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.max_len = 200
        self.position_encoder = PositionEmbeddings(config)
        self.fc1 = torch.nn.Linear(165, config.d_model)
        self.encoder = AttentionStack(config, is_decoder=False)
        self.decoder = AttentionStack(config, is_decoder=True)
        self.fc2 = torch.nn.Linear(config.d_model, config.d_probability)
        # self.normal_flow = NormalFlow(config)
        self.fc3 = torch.nn.Linear(config.d_probability, 165)

    def forward(self, p1_vectors, p2_vectors):
        seq_len = p1_vectors.shape[1]
        if seq_len <= self.max_len:
            model_output = self.direct_forward(p1_vectors, p2_vectors)
        else:
            model_output1 = self.direct_forward(p1_vectors[:, :self.max_len, :], p2_vectors[:, :self.max_len, :])
            model_output2 = self.window_forward(p1_vectors[:, seq_len - self.max_len - 2:, :],
                                                p2_vectors[:, seq_len - self.max_len - 2:, :],
                                                seq_len - self.max_len)
            model_output = torch.cat((model_output1, model_output2), 1)
        return model_output

    def direct_forward(self, p1_vectors, p2_vectors):
        seq_len = p1_vectors.shape[1]
        p1_vectors = self.fc1(p1_vectors)#.to(torch.float32))
        p2_vectors = self.fc1(p2_vectors)#.to(torch.float32))
        position_ids = torch.tensor([i for i in range(seq_len)]).cuda()
        position_embeddings = self.position_encoder(position_ids).unsqueeze(0)
        p1_vectors += position_embeddings
        p2_vectors += position_embeddings
        encoder_outputs = self.encoder(p1_vectors)
        decoder_outputs = self.decoder(encoder_outputs, p2_vectors)
        cross_attn_embeddings = self.fc2(decoder_outputs)
        # model_output = self.normal_flow(cross_attn_embeddings)
        model_output = self.fc3(cross_attn_embeddings)
        return model_output

    def window_forward(self, p1_vectors, p2_vectors, times):
        seq_len = p1_vectors.shape[1]
        cur_p2_outputs = None
        for i in tqdm(range(times)):
            cur_p1_inputs = p1_vectors[:, i:self.max_len + i + 1, :]
            cur_p2_inputs = p2_vectors[:, i:self.max_len + i + 1, :]
            cur_p2_output = self.forward(p1_vectors=cur_p1_inputs,
                                         p2_vectors=cur_p2_inputs)
            if i == 0:
                cur_p2_outputs = cur_p2_output[:, -1:, :]
            else:
                cur_p2_outputs = torch.cat((cur_p2_outputs, cur_p2_output[:, -1:, :]), 1)
        model_output = cur_p2_outputs
        return model_output

    def generate(self, inputs):
        model_output = None
        with torch.no_grad():
            seq_len = inputs.shape[1]
            for i in tqdm(range(seq_len)):
                cur_p1_inputs = inputs[:, :i + 1, :]
                if i == 0:
                    cur_p2_output = self.direct_forward(p1_vectors=cur_p1_inputs,
                                                        p2_vectors=cur_p1_inputs)
                    model_output = torch.cat((cur_p2_output, cur_p2_output), 1)
                else:
                    if cur_p1_inputs.shape[1] > self.max_len:
                        cur_p1_inputs = cur_p1_inputs[:, i+1-self.max_len:i+1, :]
                        cur_p2_inputs = model_output[:, i+1-self.max_len:i+1, :]
                    else:
                        cur_p2_inputs = model_output
                    cur_p2_output = self.direct_forward(p1_vectors=cur_p1_inputs,
                                                        p2_vectors=cur_p2_inputs)
                    model_output = torch.cat((model_output, cur_p2_output[:, -1:, :]), 1)
        return model_output[:, 1:, :]
