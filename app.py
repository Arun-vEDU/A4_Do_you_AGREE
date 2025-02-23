import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Updated hyperparameters to match the checkpoint
n_layers = 12  # Number of layers in the checkpoint
n_heads = 12   # Number of attention heads in the checkpoint
d_model = 768  # Hidden size in the checkpoint
d_ff = 768 * 4
d_k = d_v = 64
n_segments = 2
vocab_size = 60305  # Vocabulary size in the checkpoint
max_len = 1000

# Custom BERT Implementation (unchanged)
class Embedding(nn.Module):
    def __init__(self, vocab_size, max_len, n_segments, d_model, device):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long).to(self.device)
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

def get_attn_pad_mask(seq_q, seq_k, device):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1).to(device)
    return pad_attn_mask.expand(batch_size, len_q, len_k)

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, d_k, device):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(n_heads, d_model, d_k, device)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, device):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = torch.sqrt(torch.FloatTensor([d_k])).to(device)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, device):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_k
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, self.d_v * n_heads)
        self.device = device

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = ScaledDotProductAttention(self.d_k, self.device)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = nn.Linear(self.n_heads * self.d_v, self.d_model).to(self.device)(context)
        return nn.LayerNorm(self.d_model).to(self.device)(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class BERT(nn.Module):
    def __init__(self, n_layers, n_heads, d_model, d_ff, d_k, n_segments, vocab_size, max_len, device):
        super(BERT, self).__init__()
        self.embedding = Embedding(vocab_size, max_len, n_segments, d_model, device)
        self.layers = nn.ModuleList([EncoderLayer(n_heads, d_model, d_ff, d_k, device) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))
        self.device = device

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, self.device)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        h_pooled = self.activ(self.fc(output[:, 0]))
        logits_nsp = self.classifier(h_pooled)
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.norm(F.gelu(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias
        return logits_lm, logits_nsp

    def get_last_hidden_state(self, input_ids, segment_ids):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, self.device)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        return output

# Initialize the model with updated hyperparameters
custom_bert = BERT(
    n_layers=12,  # Updated to match the checkpoint
    n_heads=12,   # Updated to match the checkpoint
    d_model=768,  # Updated to match the checkpoint
    d_ff=768 * 4,
    d_k=64,
    n_segments=2,
    vocab_size=60305,  # Updated to match the checkpoint
    max_len=1000,
    device=device
).to(device)

# Load the checkpoint
custom_bert.load_state_dict(torch.load('bert_model.pth', map_location=device))
custom_bert.eval()

# SentenceBERT Model (unchanged)
def mean_pool(token_embeds, attention_mask):
    in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
    return pool

class SentenceBERT(nn.Module):
    def __init__(self, bert_model):
        super(SentenceBERT, self).__init__()
        self.bert = bert_model

    def forward(self, premise_input_ids, premise_attention_mask, hypothesis_input_ids, hypothesis_attention_mask):
        premise_segment_ids = torch.zeros_like(premise_input_ids, device=self.bert.device)
        hypothesis_segment_ids = torch.zeros_like(hypothesis_input_ids, device=self.bert.device)
        premise_embeds = self.bert.get_last_hidden_state(premise_input_ids, premise_segment_ids)
        u = mean_pool(premise_embeds, premise_attention_mask)
        hypothesis_embeds = self.bert.get_last_hidden_state(hypothesis_input_ids, hypothesis_segment_ids)
        v = mean_pool(hypothesis_embeds, hypothesis_attention_mask)
        return u, v

# Initialize SentenceBERT
sbert = SentenceBERT(custom_bert).to(device)

# Load SentenceBERT Model Weights
sbert.load_state_dict(torch.load('sbert_model.pth', map_location=device))
sbert.eval()

# Load Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prediction Function (unchanged)
def predict_nli(premise, hypothesis):
    premise_inputs = tokenizer(premise, padding=True, truncation=True, return_tensors='pt').to(device)
    hypothesis_inputs = tokenizer(hypothesis, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        u, v = sbert(
            premise_inputs['input_ids'], 
            premise_inputs['attention_mask'], 
            hypothesis_inputs['input_ids'], 
            hypothesis_inputs['attention_mask']
        )
    cosine_sim = torch.nn.functional.cosine_similarity(u, v, dim=1)
    if cosine_sim > 0.8:
        return "Entailment"
    elif cosine_sim > 0.5:
        return "Neutral"
    else:
        return "Contradiction"
# Dash App 
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("Text Similarity and NLI Prediction"), className="mb-4")]),
    dbc.Row([
        dbc.Col([html.Label("Premise:"), dcc.Input(id="premise-input", type="text", placeholder="Enter premise...", className="form-control")]),
        dbc.Col([html.Label("Hypothesis:"), dcc.Input(id="hypothesis-input", type="text", placeholder="Enter hypothesis...", className="form-control")])
    ]),
    dbc.Row([dbc.Col(html.Button("Predict", id="predict-button", className="btn btn-primary mt-3"))]),
    dbc.Row([dbc.Col(html.Div(id="output", className="mt-4"))])
])

@app.callback(
    Output("output", "children"),
    Input("predict-button", "n_clicks"),
    Input("premise-input", "value"),
    Input("hypothesis-input", "value")
)
def update_output(n_clicks, premise, hypothesis):
    if n_clicks is None:
        return ""
    if not premise or not hypothesis:
        return "Please enter both premise and hypothesis."
    try:
        label = predict_nli(premise, hypothesis)
        return html.Div([
            html.H4("Prediction:"),
            html.P(f"Premise: {premise}"),
            html.P(f"Hypothesis: {hypothesis}"),
            html.P(f"Label: {label}", style={"color": "blue", "font-weight": "bold"})
        ])
    except Exception as e:
        return html.Div(f"An error occurred: {str(e)}", style={"color": "red"})

if __name__ == "__main__":
    app.run_server(debug=True)