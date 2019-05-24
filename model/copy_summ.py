import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .attention import step_attention
from .util import len_mask
from .summ import Seq2SeqSumm, AttentionalLSTMDecoder
from . import beam_search as bs


INIT = 1e-2
class SelfAttentiveEncoder(nn.Module):

    def __init__(self, config):
        super(SelfAttentiveEncoder, self).__init__()
        self.drop = nn.Dropout(config['dropout'])
        self.ws1 = nn.Linear(config['nhid'], config['attention-unit'], bias=False)
        self.ws2 = nn.Linear(config['attention-unit'], config['attention-hops'], bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.dictionary = {"PAD": 0}
#        self.init_weights()
        self.attention_hops = config['attention-hops']

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp, outp):
        #inp:[b,len]
        #outp:[b, len, hid]
        size = outp.size()  # [bsz, len, nhid]
        compressed_embeddings = outp.contiguous().view(-1, size[2])  # [bsz*len, nhid*2]
        #transformed_inp = torch.transpose(inp, 0, 1).contiguous()  # [bsz, len]
        transformed_inp = inp.view(size[0], 1, size[1])  # [bsz, 1, len]
        concatenated_inp = [transformed_inp for i in range(self.attention_hops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, len]

        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        penalized_alphas = alphas + (
            -10000 * (concatenated_inp == self.dictionary['PAD']).float())
            # [bsz, hop, len] + [bsz, hop, len]
        alphas = self.softmax(penalized_alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas

    def init_hidden(self, bsz):
        return self.bilstm.init_hidden(bsz)

class _CopyLinear(nn.Module):
    def __init__(self, context_dim, state_dim, input_dim, bias=True):
        super().__init__()
        self._v_c = nn.Parameter(torch.Tensor(context_dim))
        self._v_s = nn.Parameter(torch.Tensor(state_dim))
        self._v_i = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._v_c, -INIT, INIT)
        init.uniform_(self._v_s, -INIT, INIT)
        init.uniform_(self._v_i, -INIT, INIT)
        if bias:
            self._b = nn.Parameter(torch.zeros(1))
        else:
            self.regiser_module(None, '_b')

    def forward(self, context, state, input_):
        output = (torch.matmul(context, self._v_c.unsqueeze(1))
                  + torch.matmul(state, self._v_s.unsqueeze(1))
                  + torch.matmul(input_, self._v_i.unsqueeze(1)))
        if self._b is not None:
            output = output + self._b.unsqueeze(0)
        return output


class CopySumm(Seq2SeqSumm):
    def __init__(self, vocab_size, emb_dim,
                 n_hidden, bidirectional, n_layer, dropout=0.0):
        super().__init__(vocab_size, emb_dim,
                         n_hidden, bidirectional, n_layer, dropout)
        self._copy = _CopyLinear(n_hidden, n_hidden, 2*emb_dim)
        self._decoder = CopyLSTMDecoder(
            self._copy, self._embedding, self._dec_lstm,
            self._attn_wq, self._projection, self._coverage, self.c_f
        )
        self.unit = 128
        self.hops = 32
        self_atte_config = {
        'dropout': dropout,
        'nhid': n_hidden,
        'attention-unit': self.unit,
        'attention-hops': self.hops,
    }
        self.art_compress = SelfAttentiveEncoder(self_atte_config)
        self.abs_compress = SelfAttentiveEncoder(self_atte_config)

        # self.uncompress = nn.Sequential(
        #     nn.Linear(n_hidden - 128, n_hidden),
        #     nn.Tanh(),
        #     nn.Linear(n_hidden, n_hidden),
        #     nn.Tanh(),
        #     nn.Linear(n_hidden, n_hidden),
        # )
    def forward(self, article, art_lens, abstract, extend_art, extend_vsize, v_size, abs_lens=None,inp_abs=None, extend_abs=None):
        #this is ueed for training abstractor
        #article:(b,sL)
        #arl_len:(b)
        #abstract:(b,tL)
        #extend_art:(b,sL) the new word is added into the vocabulary
        #extend_vsize: nubmer (the size os extend vocabulary with new word)
        if abs_lens is not None and extend_abs is not None:
            source_abs, init_abs = self.encode(inp_abs, abs_lens)
            mask_abs = len_mask(abs_lens, source_abs.device).unsqueeze(-2)
            #mask_abs = None
            compressed_abs, abs_a = self.abs_compress.forward(inp_abs, source_abs) #(hops, hid)

            #attention_abs = self.uncompress(inp_abs,attention_abs)
            logit_abs, score_abs, coverage_abs = self._decoder(
            (compressed_abs, source_abs, mask_abs, extend_abs, extend_vsize, v_size),
            init_abs, abstract)

        source_art, init_dec_states = self.encode(article, art_lens)
        #attention (b, sL, h)
        # init_dec_states= [[init_h:(b,h), init_c:(b,h)], init_attn_out:(b, en_dim)]
        compressed_art, art_a = self.art_compress.forward(article, source_art)
        #attention = self.uncompress(attention)

        mask = len_mask(art_lens, source_art.device).unsqueeze(-2)
        #mask = None
        logit, score_art, coverage_art = self._decoder(
            (compressed_art, source_art, mask, extend_art, extend_vsize, v_size),
            init_dec_states, abstract
        )
        return (logit, art_a, compressed_art, score_art, coverage_art), (logit_abs, abs_a, compressed_abs, score_abs, coverage_abs)

    def batch_decode(self, article, art_lens, extend_art, extend_vsize, v_size, go, eos, unk, max_len):
        #this is used for training full model
        """ greedy decode support batching"""
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        source_attention, init_dec_states = self.encode(article, art_lens)
        attention , _ = self.art_compress.forward(article, source_attention)


        mask = len_mask(art_lens, source_attention.device).unsqueeze(-2)
        #mask = None
        attention = (attention, source_attention, mask, extend_art, extend_vsize, v_size)
        tok = torch.LongTensor([go]*batch_size).to(article.device)
        outputs = []
        attns = []
        s_len = source_attention.size(1)
        coverage = torch.zeros(batch_size, s_len).to(source_attention.device)
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score, coverage = self._decoder.decode_step(
                tok, states, attention, coverage)
            attns.append(attn_score)
            outputs.append(tok[:, 0].clone())
            tok.masked_fill_(tok >= vsize, unk)
        return outputs, attns

    def decode(self, article, extend_art, extend_vsize, v_size, go, eos, unk, max_len):
        vsize = self._embedding.num_embeddings
        source_attention, init_dec_states = self.encode(article)
        com_attention, _ = self.art_compress.forward(article, source_attention)

        attention = (com_attention, source_attention, None, extend_art, extend_vsize,v_size)
        tok = torch.LongTensor([go]).to(article.device)
        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score = self._decoder.decode_step(
                tok, states, attention)
            if tok[0, 0].item() == eos:
                break
            outputs.append(tok[0, 0].item())
            attns.append(attn_score.squeeze(0))
            if tok[0, 0].item() >= vsize:
                tok[0, 0] = unk
        return outputs, attns

    def batched_beamsearch(self, article, art_lens,
                           extend_art, extend_vsize, v_size,
                           go, eos, unk, max_len, beam_size, diverse=1.0):
        #used for decoding with beam
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        source_attention, init_dec_states = self.encode(article, art_lens)
        com_attention, _ = self.art_compress.forward(article, source_attention)
        mask = len_mask(art_lens, source_attention.device).unsqueeze(-2)
        all_attention = (com_attention, source_attention, mask, extend_art, extend_vsize, v_size)
        s_len = source_attention.size(1)
        coverage = torch.zeros(batch_size, s_len).to(source_attention.device)
        coverage_all = coverage
        attention = all_attention
        (h, c), prev = init_dec_states

        all_beams = [bs.init_beam(go, (h[:, i, :], c[:, i, :], prev[i]))
                     for i in range(batch_size)]  #[[_Hypothesis,]]
        finished_beams = [[] for _ in range(batch_size)] # [batch_size]
        outputs = [None for _ in range(batch_size)] # [batch_size]
        for t in range(max_len):
            toks = []
            all_states = []
            for beam in filter(bool, all_beams):
                token, states = bs.pack_beam(beam, article.device)
                toks.append(token)
                all_states.append(states)
            token = torch.stack(toks, dim=1)
            states = ((torch.stack([h for (h, _), _ in all_states], dim=2),
                       torch.stack([c for (_, c), _ in all_states], dim=2)),
                      torch.stack([prev for _, prev in all_states], dim=1))
            token.masked_fill_(token >= vsize, unk)

            topk, lp, states, attn_score, coverage = self._decoder.topk_step(
                token, states, attention, beam_size, coverage)
            ind_all = [j for j, o in enumerate(outputs) if o is None]
            ind_all = torch.LongTensor(ind_all).to(source_attention.device)
            coverage_all.index_add_(0, ind_all, coverage)
            batch_i = 0

            for i, (beam, finished) in enumerate(zip(all_beams,
                                                     finished_beams)):
                if not beam:
                    continue
                finished, new_beam = bs.next_search_beam(
                    beam, beam_size, finished, eos,
                    topk[:, batch_i, :], lp[:, batch_i, :],
                    (states[0][0][:, :, batch_i, :],
                     states[0][1][:, :, batch_i, :],
                     states[1][:, batch_i, :]),
                    attn_score[:, batch_i, :],
                    diverse
                )
                batch_i += 1

                if len(finished) >= beam_size:
                    all_beams[i] = []
                    outputs[i] = finished[:beam_size]
                    # exclude finished inputs
                    (com_attention, source_attention, mask, extend_art, extend_vsize, v_size) = all_attention
                    coverage_ind = coverage_all
                    masks = [mask[j] for j, o in enumerate(outputs) if o is None]
                    ind = [j for j, o in enumerate(outputs) if o is None]
                    ind = torch.LongTensor(ind).to(source_attention.device)
                    source_attention, com_attention, extend_art, coverage_ind = map(
                        lambda v: v.index_select(dim=0, index=ind),
                        [source_attention, com_attention, extend_art, coverage_ind]
                    )

                    if masks:
                        #print(len(masks), masks[0])
                        mask = torch.stack(masks, dim=0)
                    else:
                        mask = None
                    coverage = coverage_ind
                    attention = (
                        com_attention, source_attention, mask, extend_art, extend_vsize, v_size)
                else:
                    all_beams[i] = new_beam
                    finished_beams[i] = finished
            if all(outputs):
                break
        else:
            for i, (o, f, b) in enumerate(zip(outputs,
                                              finished_beams, all_beams)):
                if o is None:
                    outputs[i] = (f+b)[:beam_size]
        return outputs


class CopyLSTMDecoder(AttentionalLSTMDecoder):
    def __init__(self, copy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._copy = copy


    def _step(self, tok, states, attention, coverage):
        #coverage: (b, ext_vsize)
        prev_states, prev_out = states
        lstm_in = torch.cat(
            [self._embedding(tok).squeeze(1), prev_out],
            dim=1
        )
        states = self._lstm(lstm_in, prev_states)
        lstm_out = states[0][-1]
        query = torch.mm(lstm_out, self._attn_w)  # (b, emb_dim) * (emb_dim, hidden)
        compress_attention, source_attention, attn_mask, extend_src, extend_vsize, v_size = attention

        # context, _, _ = step_attention(
        #     query, compress_attention, compress_attention)

        context, score, coverage = step_attention(
            query, source_attention, source_attention, attn_mask, coverage, self.c_f)
        copy_mask = torch.ge(extend_src, v_size).type_as(score).to(score.device)


        dec_out = self._projection(torch.cat([lstm_out, context], dim=1))  #(b, emb_dim)

        # extend generation prob to extended vocabulary
        gen_prob = self._compute_gen_prob(dec_out, extend_vsize)
        # compute the probabilty of each copying
        copy_prob = torch.sigmoid(self._copy(context, states[0][-1], lstm_in))
        # add the copy prob to existing vocab distribution
        #print(extend_src.size(), score.size())
        lp = torch.log(
            ((-copy_prob + 1) * gen_prob
            ).scatter_add_(
                dim=1,
                index=extend_src.expand_as(score),
                src=score * copy_mask * copy_prob)
            + 1e-8)  # numerical stability for log
        return lp, (states, dec_out), score, coverage


    def topk_step(self, tok, states, attention, k, coverage):
        """tok:[BB, B], states ([L, BB, B, D]*2, [BB, B, D])"""
        (h, c), prev_out = states

        # lstm is not bemable
        nl, _, _, d = h.size()
        beam, batch = tok.size()
        lstm_in_beamable = torch.cat(
            [self._embedding(tok), prev_out], dim=-1)
        lstm_in = lstm_in_beamable.contiguous().view(beam*batch, -1)
        prev_states = (h.contiguous().view(nl, -1, d),
                       c.contiguous().view(nl, -1, d))
        h, c = self._lstm(lstm_in, prev_states)
        states = (h.contiguous().view(nl, beam, batch, -1),
                  c.contiguous().view(nl, beam, batch, -1))
        lstm_out = states[0][-1]

        # attention is beamable
        query = torch.matmul(lstm_out, self._attn_w)
        compress_attention, source_attention, attn_mask, extend_src, extend_vsize, v_size = attention
        #attention, attn_mask, extend_src, extend_vsize = attention
        context, score, coverage = step_attention(
            query, source_attention, source_attention, attn_mask, coverage, self.c_f)
        #copy_mask = torch.ge(extend_src, v_size).type_as(score).to(score.device)

        # context, _, _ = step_attention(
        #     query, compress_attention, compress_attention)

        #context, score = step_attention(
            #query, attention, attention, attn_mask)
        dec_out = self._projection(torch.cat([lstm_out, context], dim=-1))

        # copy mechanism is not beamable
        gen_prob = self._compute_gen_prob(
            dec_out.contiguous().view(batch*beam, -1), extend_vsize)
        copy_prob = torch.sigmoid(
            self._copy(context, lstm_out, lstm_in_beamable)
        ).contiguous().view(-1, 1)
        lp = torch.log(
            ((-copy_prob + 1) * gen_prob
            ).scatter_add_(
                dim=1,
                index=extend_src.expand_as(score).contiguous().view(beam*batch, -1),
                src=score.contiguous().view(beam*batch, -1) * copy_prob
        ) + 1e-8).contiguous().view(beam, batch, -1)

        k_lp, k_tok = lp.topk(k=k, dim=-1)
        return k_tok, k_lp, (states, dec_out), score, coverage

    def _compute_gen_prob(self, dec_out, extend_vsize, eps=1e-6):
        logit = torch.mm(dec_out, self._embedding.weight.t()) #(b, v_size)== (b,enc_dim) * (v_size, enc_dim)
        bsize, vsize = logit.size()
        if extend_vsize > vsize:
            ext_logit = torch.Tensor(bsize, extend_vsize-vsize
                                    ).to(logit.device)
            ext_logit.fill_(eps)
            gen_logit = torch.cat([logit, ext_logit], dim=1)
        else:
            gen_logit = logit
        gen_prob = F.softmax(gen_logit, dim=-1)
        return gen_prob

    def _compute_copy_activation(self, context, state, input_, score):
        copy = self._copy(context, state, input_) * score
        return copy
