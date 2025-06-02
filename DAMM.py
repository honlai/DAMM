import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast, GradScaler

import torchvision
import torchvision.models as models
from torchvision import transforms
from torchvision.models import ResNet101_Weights

from PIL import Image

import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

from collections import Counter
from tqdm import tqdm
import time
import re
import json
from nltk.tokenize import word_tokenize
import numpy as np
import open_clip

import matplotlib.pyplot as plt
import cv2

class Vocabulary:
    def __init__(self, save_file=None):
        self.words = []
        self.word2idx = {}
        self.word_frequencies = []
        if save_file is not None:
            self.load(save_file)

    def __len__(self):
        return len(self.words)

    def _clean_word(self, word):
        """ Helper function to remove special characters from a word. """
        return re.sub(r'[^a-zA-Z0-9]', '', word)

    def build_from_chat_json(self, chat_json_path, min_freq=1):
        """For LLaVA/CC3M chat.json"""
        with open(chat_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sentences = []
        for entry in data:
            for convo in entry["conversations"]:
                text = convo["value"]
                clean_text = text.replace("<image>", "").strip()
                sentences.append(clean_text)

        self.build(sentences, min_freq=min_freq)
        
    def build(self, sentences, min_freq=1, use_subword=False):
        """ Build the vocabulary and compute the frequency of each word. """
        word_counts = Counter()
        
        # Count word frequencies
        for sentence in tqdm(sentences, desc="Building vocabulary"):
            if sentence:
                tokens = [self._clean_word(word).lower() for word in word_tokenize(sentence) if self._clean_word(word)]
                word_counts.update(tokens)

        # Apply minimum frequency threshold
        word_counts = {word: freq for word, freq in word_counts.items() if freq >= min_freq}
        unique_word_count = len(word_counts)

        # Add special tokens
        special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
        for i, token in enumerate(special_tokens):
            self.words.append(token)
            self.word2idx[token] = i
            self.word_frequencies.append(1.0 if token != '<PAD>' else 0.0)

        # Sort by frequency and add words
        sorted_word_counts = Counter(word_counts).most_common()
        for idx, (word, frequency) in enumerate(sorted_word_counts, start=4):
            self.words.append(word)
            self.word2idx[word] = idx
            self.word_frequencies.append(frequency)
        # Convert frequencies to numpy array, normalize, and log-transform
        self.word_frequencies = np.array(self.word_frequencies, dtype=np.float32) + 1e-10
        self.word_frequencies /= np.sum(self.word_frequencies)
        self.word_frequencies = np.log1p(self.word_frequencies)
        self.word_frequencies -= np.max(self.word_frequencies)
        
    def encode(self, sentence):
        """ Convert a sentence to indices, adding special tokens. """
        words = [self._clean_word(word).lower() for word in word_tokenize(sentence) if self._clean_word(word)]
        word_idxs = [self.word2idx.get('<START>')]  # Add <START>
        word_idxs.extend([self.word2idx.get(w, self.word2idx['<UNK>']) for w in words])  # Convert words to indices
        word_idxs.append(self.word2idx.get('<END>'))  # Add <END>
        return word_idxs, len(word_idxs)

    def decode(self, idxs, skip_special_tokens=True):
        """ Convert indices back to a sentence. """
        words = []
        for idx in idxs:
            if idx >= len(self.words):
                print(f"Warning: Index {idx} is out of vocabulary bounds.")
                continue
            word = self.words[idx]
            if skip_special_tokens:
                if word == '<END>':
                    break
                if word != '<PAD>':
                    words.append(word)
            else:
                words.append(word)
        return " ".join(words).strip()

    def save(self, save_file):
        """ Save vocabulary to a compressed .npz file. """
        np.savez(save_file, words=self.words, word2idx=self.word2idx, frequencies=self.word_frequencies)

    def load(self, save_file):
        """ Load vocabulary from a compressed file. """
        if not os.path.exists(save_file):
            raise FileNotFoundError(f"File {save_file} does not exist.")
        data = np.load(save_file, allow_pickle=True)
        self.words = list(data['words'])
        self.word2idx = data['word2idx'].item()
        self.word_frequencies = data['frequencies']

    def add_sentence(self, sentence):
        """ Dynamically add a new sentence to the vocabulary. """
        tokens = [self._clean_word(word).lower() for word in word_tokenize(sentence) if self._clean_word(word)]
        new_counts = Counter(tokens)
        for word, freq in new_counts.items():
            if word not in self.word2idx:
                idx = len(self.words)
                self.words.append(word)
                self.word2idx[word] = idx
                self.word_frequencies = np.append(self.word_frequencies, freq)
            else:
                self.word_frequencies[self.word2idx[word]] += freq

class LLaVAQADataset(Dataset):
    def __init__(self, json_path, image_dir, vocab, image_transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.image_dir = image_dir
        self.vocab = vocab
        if image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.image_transform = image_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_id = item['image']
        question = item['conversations'][0]['value'].replace('<image>', '').strip()
        answer = item['conversations'][1]['value'].strip()

        image_path = os.path.join(self.image_dir, f"{image_id}")
        image = Image.open(image_path).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)
        question_idxs, _ = self.vocab.encode(question)
        answer_idxs, _ = self.vocab.encode(answer)
        batch = {
            'image': image,
            'question_idxs': question_idxs,
            'caption_idxs': answer_idxs
        }
        
        return self.collate_fn(batch)
    def collate_fn(self, batch, max_q_len=64, max_a_len=128):
        images = batch['image']
        questions = batch['question_idxs'][:max_q_len]
        captions = batch['caption_idxs'][:max_a_len]

        questions_pad = np.pad(questions, [0,max_q_len-len(questions)], mode='constant')
        captions_pad = np.pad(captions, [0,max_a_len-len(captions)], mode='constant')

        return {
            'image': images,
            'question_idxs': torch.tensor(questions_pad, dtype=torch.long),
            'caption_idxs': torch.tensor(captions_pad, dtype=torch.long)
        }
    
# 1. Image Feature Encoder
class ImageFeatureEncoder(nn.Module):
    def __init__(self, image_feat_dim=2048, freeze_backbone=False):
        super().__init__()
        model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(model.children())[:-2])
        self.proj = nn.Conv2d(2048, image_feat_dim, kernel_size=1) if image_feat_dim != 2048 else nn.Identity()
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        feats = self.backbone(x)
        return self.proj(feats)

class CLIPResNetImageFeatureEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "RN101",
        pretrained: str = "openai",
        target_layer: str = "layer4",
        freeze_backbone: bool = True,
    ):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        visual = model.visual
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        visual.to(self.device)  

        self.target_layer = target_layer

        # Use submodules directly from CLIP ModifiedResNet
        self.stem = visual.stem
        self.avgpool = visual.avgpool
        self.layer1 = visual.layer1
        self.layer2 = visual.layer2
        self.layer3 = visual.layer3
        self.layer4 = visual.layer4

        if freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        # x = self.avgpool(x)

        x = self.layer1(x)
        if self.target_layer == "layer1":
            return x

        x = self.layer2(x)
        if self.target_layer == "layer2":
            return x

        x = self.layer3(x)
        if self.target_layer == "layer3":
            return x

        x = self.layer4(x)
        return x  # default to layer4


  
# 2. Question Encoder
class BiLSTMQuestionEncoder(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    
    Useage:
        encoder = BiLSTMQuestionEncoder(vocab_size=30522, embedding_dim=300, hidden_dim=512)
        input_ids = torch.tensor([[3, 10, 20, 5, 0, 0], [7, 8, 9, 0, 0, 0]])  # [B, T]
        lengths = torch.tensor([4, 3])  # real sequence lengths

        output = encoder(input_ids, lengths)  # [B, T, 1024]
    """
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=512, num_layers=1, dropout=0.3, use_pretrained_embeddings=None):
        super(BiLSTMQuestionEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if use_pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(use_pretrained_embeddings)
            self.embedding.weight.requires_grad = False  # Optional: freeze embeddings

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, input_ids, lengths):
        """
        input_ids: [batch_size, seq_len] - question word indices
        lengths: [batch_size] - actual lengths before padding
        """

        # [B, T, D]
        embedded = self.embedding(input_ids)

        # Pack padded sequence for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed)
        # Unpack
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)  # output: [B, T, 2*H]

        return output  # full sequence hidden states: [B, T, 2*hidden_dim]

# 3. Attention
class QuestionAttention(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
        
    Useage:
        decoder_hidden = torch.randn(batch_size, 512)             # [B, H_dec]
        question_outputs = torch.randn(batch_size, seq_len, 1024) # [B, T, H_q]
        question_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])  # [B, T]

        attn_layer = QuestionAttention(decoder_hidden_dim=512, question_hidden_dim=1024)
        context_vec, attn_weights = attn_layer(decoder_hidden, question_outputs, question_mask)
    """
    def __init__(self, decoder_hidden_dim, question_hidden_dim, attention_dim=256):
        super(QuestionAttention, self).__init__()

        self.decoder_proj = nn.Linear(decoder_hidden_dim, attention_dim)
        self.question_proj = nn.Linear(question_hidden_dim, attention_dim)
        self.attn_score = nn.Linear(attention_dim, 1)

    def forward(self, decoder_hidden, question_outputs, question_mask=None):
        """
        decoder_hidden: [B, H_dec] (current decoder hidden state)
        question_outputs: [B, T, H_q] (BiLSTM hidden outputs for question)
        question_mask: [B, T] 1 for real tokens, 0 for padding
        """

        B, T, H_q = question_outputs.size()

        # Expand decoder hidden state to [B, T, H_dec]
        decoder_expanded = decoder_hidden.unsqueeze(1).expand(-1, T, -1)

        # Compute score
        proj_dec = self.decoder_proj(decoder_expanded)      # [B, T, A]
        proj_q = self.question_proj(question_outputs)        # [B, T, A]
        combined = torch.tanh(proj_dec + proj_q)             # [B, T, A]
        scores = self.attn_score(combined).squeeze(-1)       # [B, T]

        if question_mask is not None:
            scores = scores.masked_fill(question_mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=1)              # [B, T]
        context = torch.bmm(attn_weights.unsqueeze(1), question_outputs)  # [B, 1, H_q]
        context = context.squeeze(1)                         # [B, H_q]

        return context, attn_weights
    
# question_outputs = encoder(question_ids, lengths)  # [B, T, H_q]
# context_vec, attn_weights = question_attention(decoder_hidden, question_outputs, question_mask)
class ImageAttention(nn.Module):
    def __init__(self, decoder_hidden_dim, image_feat_dim, attention_dim=256):
        super(ImageAttention, self).__init__()

        self.decoder_proj = nn.Linear(decoder_hidden_dim, attention_dim)
        self.image_proj = nn.Linear(image_feat_dim, attention_dim)
        self.attn_score = nn.Linear(attention_dim, 1)

    def forward(self, decoder_hidden, image_feats):
        """
        decoder_hidden: [B, H_dec]
        image_feats: [B, C, H, W] (e.g., [B, 2048, 14, 14])
        """

        B, C, H, W = image_feats.size()
        N = H * W

        # Flatten image features: [B, C, H, W] -> [B, N, C]
        image_feats = image_feats.view(B, C, N).permute(0, 2, 1)  # [B, N, C]

        # Expand decoder hidden: [B, H_dec] -> [B, N, H_dec]
        decoder_expanded = decoder_hidden.unsqueeze(1).expand(-1, N, -1)  # [B, N, H_dec]

        # Apply projections
        proj_dec = self.decoder_proj(decoder_expanded)     # [B, N, A]
        proj_img = self.image_proj(image_feats)            # [B, N, A]
        combined = torch.tanh(proj_dec + proj_img)         # [B, N, A]
        scores = self.attn_score(combined).squeeze(-1)     # [B, N]

        attn_weights = F.softmax(scores, dim=1)            # [B, N]

        # Weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), image_feats)  # [B, 1, C]
        context = context.squeeze(1)  # [B, C]

        return context, attn_weights # [B, H*W]

class GatedFusionModule(nn.Module):
    """
    Performs modality-aware gated fusion by projecting both image_context and
    question_context into a shared semantic space, where the decoder hidden state
    dynamically controls the weighting between visual and textual modalities.
    """
    def __init__(self, image_feat_dim, question_feat_dim, decoder_hidden_dim, fused_dim):
        super().__init__()
        
        self.image_proj = nn.Linear(image_feat_dim, fused_dim)
        self.question_proj = nn.Linear(question_feat_dim, fused_dim)
        self.gate_proj = nn.Linear(decoder_hidden_dim, 2)

        self.fused_dim = fused_dim

    def forward(self, image_context, question_context, decoder_hidden, record_gate_list=None):
        """
        image_context: [B, C_img]
        question_context: [B, C_q]
        decoder_hidden: [B, H_dec]
        """
        img_proj = self.image_proj(image_context)          # [B, D]
        ques_proj = self.question_proj(question_context)   # [B, D]

        gate_logits = self.gate_proj(decoder_hidden)       # [B, 2]
        gate_weights = F.softmax(gate_logits, dim=-1)      # [B, 2]
        gate_img, gate_ques = gate_weights[:, 0:1], gate_weights[:, 1:2]

        fused_context = gate_img * img_proj + gate_ques * ques_proj  # [B, D]

        if record_gate_list is not None:
            record_gate_list.append(gate_weights.detach().cpu())

        return fused_context

class MultimodalDecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, decoder_hidden_dim, image_feat_dim, question_feat_dim, fused_dim = None, dropout=0.3):
        super(MultimodalDecoderLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # Gated fusion
        if fused_dim is None:
            fused_dim = (image_feat_dim + question_feat_dim) // 2
            
        self.fusion_module = GatedFusionModule(
            image_feat_dim=image_feat_dim,
            question_feat_dim=question_feat_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            fused_dim=fused_dim
        )

        # LSTM input
        lstm_input_dim = embedding_dim + fused_dim
        self.lstm = nn.LSTMCell(lstm_input_dim, decoder_hidden_dim)

        self.output_proj = nn.Linear(decoder_hidden_dim, vocab_size)

        # Initial hidden state from image mean
        self.init_h = nn.Linear(image_feat_dim, decoder_hidden_dim)
        self.init_c = nn.Linear(image_feat_dim, decoder_hidden_dim)

    def forward_step(self, prev_word_ids, prev_hidden, prev_cell, image_context, question_context, record_gate_list=None):
        """
        prev_word_ids: [B]
        prev_hidden, prev_cell: [B, H]
        image_context: [B, C_img]
        question_context: [B, C_q]
        """
        # Embed word
        word_emb = self.embedding(prev_word_ids)  # [B, E]
        word_emb = self.dropout(word_emb)
        # New Gated fusion
        fused_context = self.fusion_module(image_context, question_context, prev_hidden, record_gate_list)  # [B, D]
        
        # LSTM input
        lstm_input = torch.cat([word_emb, fused_context], dim=1)  # [B, E + D]
        h_t, c_t = self.lstm(lstm_input, (prev_hidden, prev_cell))
        logits = self.output_proj(h_t)  # [B, vocab_size]
        
        return logits, h_t, c_t

    def init_hidden_state(self, image_feats):
        """
        Args:
            image_feats: [B, C, H, W] - CNN feature map
        Returns:
            h_0, c_0: [B, H_dec]
        """
        B, C, H, W = image_feats.size()
        mean_feats = image_feats.view(B, C, -1).mean(dim=2)  # [B, C]
        h_0 = self.init_h(mean_feats)
        c_0 = self.init_c(mean_feats)
        return h_0, c_0

class MultimodalVQAModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim=512,
                 decoder_hidden_dim=768,
                 image_feat_dim=2048,
                 question_hidden_dim=768,
                 attention_dim=512):
        super(MultimodalVQAModel, self).__init__()

        self.image_encoder = CLIPResNetImageFeatureEncoder()

        self.question_encoder = BiLSTMQuestionEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=question_hidden_dim
        )

        self.image_attention = ImageAttention(decoder_hidden_dim, image_feat_dim, attention_dim)
        self.question_attention = QuestionAttention(decoder_hidden_dim, question_hidden_dim * 2, attention_dim)

        self.decoder = MultimodalDecoderLSTM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            image_feat_dim=image_feat_dim,
            question_feat_dim=question_hidden_dim * 2
        )
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, question_ids, question_lengths, target_ids, teacher_forcing=True):
        """
        image: [B, 3, H, W]
        question_ids: [B, T_q]
        question_lengths: [B]
        target_ids: [B, T_ans]
        """
        B, T_ans = target_ids.size()
        device = image.device

        with torch.no_grad():
            image_feats = self.image_encoder(image)  # [B, C, H, W]
        question_outputs = self.question_encoder(question_ids, question_lengths)  # [B, T_q, 2*H_q]

        h_t, c_t = self.decoder.init_hidden_state(image_feats)

        outputs = []
        input_token = target_ids[:, 0]  # <BOS>

        for t in range(1, T_ans):
            img_ctx, _ = self.image_attention(h_t, image_feats)
            q_ctx, _ = self.question_attention(h_t, question_outputs)

            logits, h_t, c_t = self.decoder.forward_step(input_token, h_t, c_t, img_ctx, q_ctx)
            outputs.append(logits.unsqueeze(1))

            input_token = target_ids[:, t] if teacher_forcing else logits.argmax(dim=1)

        outputs = torch.cat(outputs, dim=1)
        return outputs

    def generate(self, image, question_ids, question_lengths, max_len=20, bos_token_id=1, eos_token_id=2, return_log=False):
        B = image.size(0)
        assert B == 1, "generate() currently supports batch_size = 1 only."
        device = image.device

        image_feats = self.image_encoder(image)  # [1, C, H, W]
        question_outputs = self.question_encoder(question_ids, question_lengths)  # [1, T_q, 2H_q]

        h_t, c_t = self.decoder.init_hidden_state(image_feats)

        input_token = torch.full((B,), bos_token_id, dtype=torch.long, device=device)
        generated = [input_token.unsqueeze(1)]

        ended = torch.zeros(B, dtype=torch.bool, device=device)

        image_attn_log = []
        question_attn_log = []
        if return_log:
            gate_log = []
        else:
            gate_log = None
        
        for _ in range(max_len - 1):
            img_ctx, img_alpha = self.image_attention(h_t, image_feats)
            q_ctx, q_alpha = self.question_attention(h_t, question_outputs)

            logits, h_t, c_t = self.decoder.forward_step(input_token, h_t, c_t, img_ctx, q_ctx, gate_log)

            next_token = logits.argmax(dim=1)
            next_token = next_token.masked_fill(ended, eos_token_id)
            ended |= (next_token == eos_token_id)

            generated.append(next_token.unsqueeze(1))
            input_token = next_token

            if return_log:
                image_attn_log.append(img_alpha.detach().cpu())        # [1, H×W]
                question_attn_log.append(q_alpha.detach().cpu())       # [1, T_q]

            if ended.all():
                break

        gen_ids = torch.cat(generated, dim=1)  # [1, T_gen]

        if return_log:
            return gen_ids, image_attn_log, question_attn_log, gate_log
        else:
            return gen_ids

def get_caption_lengths(captions):
    eos_token_id = 2
    caption_lengths = []
    for cap in captions:
        end_pos = (cap == eos_token_id).nonzero(as_tuple=True)[0]
        length = end_pos[0].item() + 1 if len(end_pos) > 0 else len(cap) 
        caption_lengths.append(length)
    caption_lengths = torch.tensor(caption_lengths)
    return caption_lengths

def compute_loss(logits, target_ids, image_attn_weights=None, question_attn_weights=None, lambda_img=1.0, lambda_ques=0.0):
    """
    logits: [B, T, V]
    target_ids: [B, T]
    image_attn_weights: list of [B, H×W]
    question_attn_weights: list of [B, T_q]
    """
    B, T, V = logits.size()
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # assume 0 is PAD

    logits = logits.view(B * T, V)
    targets = target_ids[:, 1:].contiguous().view(-1)  # shift target by 1

    ce_loss = loss_fn(logits, targets)

    attn_loss = 0.0

    if image_attn_weights is not None and lambda_img > 0:
        # List of [B, H×W] → Tensor [B, T, H×W]
        attn_tensor = torch.stack(image_attn_weights, dim=1)
        coverage = attn_tensor.sum(dim=1)  # [B, H×W]
        attn_reg = ((1.0 - coverage) ** 2).mean()
        attn_loss += lambda_img * attn_reg

    if question_attn_weights is not None and lambda_ques > 0:
        attn_tensor = torch.stack(question_attn_weights, dim=1)
        coverage = attn_tensor.sum(dim=1)
        attn_reg = ((1.0 - coverage) ** 2).mean()
        attn_loss += lambda_ques * attn_reg

    return ce_loss + attn_loss, ce_loss.item(), attn_loss.item()

def train_epoch(model, dataloader, optimizer, device, tokenizer, epoch, lambda_img=1.0, lambda_ques=0.0):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_att = 0.0
    loss_hist = []
    
    for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        images = batch['image'].to(device)                      # [B, 3, H, W]
        questions = batch['question_idxs'].to(device)           # [B, T_q]
        captions = batch['caption_idxs'].to(device)             # [B, T_ans]

        question_lengths = (questions != 0).sum(dim=1)          # [B]

        optimizer.zero_grad()

        # forward with teacher forcing
        image_feats = model.image_encoder(images)
        question_outputs = model.question_encoder(questions, question_lengths)
        h_t, c_t = model.decoder.init_hidden_state(image_feats)

        input_token = captions[:, 0]
        outputs = []
        img_attns = []
        ques_attns = []

        for t in range(1, captions.size(1)):
            img_ctx, alpha_img = model.image_attention(h_t, image_feats)
            q_ctx, alpha_q = model.question_attention(h_t, question_outputs)

            logits, h_t, c_t = model.decoder.forward_step(input_token, h_t, c_t, img_ctx, q_ctx)
            outputs.append(logits.unsqueeze(1))

            img_attns.append(alpha_img)
            ques_attns.append(alpha_q)

            input_token = captions[:, t]

        outputs = torch.cat(outputs, dim=1)  # [B, T-1, V]
        loss, ce, attn = compute_loss(outputs, captions, img_attns, ques_attns,
                                      lambda_img=lambda_img, lambda_ques=lambda_ques)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce
        total_att += attn
        avg_loss = total_loss / (i + 1)
        loss_hist.append(avg_loss)

    avg_ce = total_ce / len(dataloader)
    avg_att = total_att / len(dataloader)

    print(f"Epoch {epoch} — Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, Attn Reg: {avg_att:.4f})")
    return loss_hist

def train_main(train_dataloader, tokenizer, vocab_size, config=None, pre_ckpt=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = {
        'embedding_dim': 512,
        'decoder_hidden_dim': 768,
        'question_hidden_dim': 768,
        'image_feat_dim': 2048,
        'attention_dim': 512,
        'dropout': 0.3,
        'learning_rate': 1e-4,
        'epochs': 10,
        'lambda_img': 1.0,
        'lambda_ques': 1.0,
        'grad_clip': 5.0,
        'save_path': './checkpoints'
    }

    if config:
        cfg.update(config)

    model = MultimodalVQAModel(
        vocab_size=vocab_size,
        embedding_dim=cfg['embedding_dim'],
        decoder_hidden_dim=cfg['decoder_hidden_dim'],
        image_feat_dim=cfg['image_feat_dim'],
        question_hidden_dim=cfg['question_hidden_dim'],
        attention_dim=cfg['attention_dim']
    ).to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg['learning_rate'])

    opt_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    start_epoch = 1
    loss_hist = []

    # Load checkpoint if exists
    start_epoch, loss_hist = load_model(
        model=model,
        optimizer=optimizer,
        opt_scheduler=opt_scheduler,
        ckpt_path=pre_ckpt,
        device=device
    )

    os.makedirs(cfg['save_path'], exist_ok=True)

    for epoch in range(start_epoch, cfg['epochs'] + 1):
        loss_hist = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            tokenizer=tokenizer,
            epoch=epoch,
            lambda_img=cfg['lambda_img'],
            lambda_ques=cfg['lambda_ques']
        )

        opt_scheduler.step()

        # Save checkpoint
        save_model(
            model=model,
            optimizer=optimizer,
            opt_scheduler=opt_scheduler,
            epoch=epoch,
            loss_hist=loss_hist,
            save_path=cfg['save_path']
        )
    return

def save_model(model, optimizer, opt_scheduler, epoch, loss_hist, save_path):
    """
    Save VQA model (excluding image encoder) and optimizer/scheduler states.

    Args:
        model: MultimodalVQAModel instance
        optimizer: optimizer used during training
        opt_scheduler: learning rate scheduler
        epoch: current epoch number (int)
        loss_hist: list or dict tracking loss history
        save_path: directory to save model checkpoint
    """
    # Create checkpoint dictionary
    ckpt = {
        'epoch': epoch,
        'question_encoder': model.question_encoder.state_dict(),
        'image_attention': model.image_attention.state_dict(),
        'question_attention': model.question_attention.state_dict(),
        'decoder': model.decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_scheduler_state_dict': opt_scheduler.state_dict(),
        'loss_hist': loss_hist
    }

    # Generate file path
    os.makedirs(save_path, exist_ok=True)
    ckpt_path = os.path.join(save_path, f"damm_epoch{epoch}.pt")

    # Save checkpoint
    torch.save(ckpt, ckpt_path)
    print(f"[INFO] Checkpoint saved: {ckpt_path}")
    return

def load_model(model, optimizer, opt_scheduler, ckpt_path, device='cpu'):
    """
    Load model components and optimizer states from a checkpoint file.

    Args:
        model: MultimodalVQAModel instance
        optimizer: optimizer instance
        opt_scheduler: learning rate scheduler
        ckpt_path: path to the checkpoint file
        device: 'cpu' or torch.device
    Returns:
        start_epoch: int, the epoch to resume from
        loss_hist: list, previously recorded loss history
    """
    if ckpt_path and os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)

        model.question_encoder.load_state_dict(checkpoint['question_encoder'])
        model.image_attention.load_state_dict(checkpoint['image_attention'])
        model.question_attention.load_state_dict(checkpoint['question_attention'])
        model.decoder.load_state_dict(checkpoint['decoder'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        opt_scheduler.load_state_dict(checkpoint.get('optimizer_scheduler_state_dict', {}))

        loss_hist = checkpoint.get('loss_hist', [])
        start_epoch = checkpoint.get('epoch', 1)

        print(f"[INFO] Resumed from checkpoint: {ckpt_path} (epoch {start_epoch})")
        return start_epoch, loss_hist
    else:
        print(f"[WARNING] Checkpoint not found: {ckpt_path}")
        return 1, []

def eval_main(model, dataloader, tokenizer, device, lambda_img=1.0, lambda_ques=0.0, save_outputs=False):
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_att = 0.0
    count = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            questions = batch['question_idxs'].to(device)
            captions = batch['caption_idxs'].to(device)
            question_lengths = (questions != 0).sum(dim=1)

            B = images.size(0)
            image_feats = model.image_encoder(images)
            question_outputs = model.question_encoder(questions, question_lengths)
            h_t, c_t = model.decoder.init_hidden_state(image_feats)

            input_token = captions[:, 0]
            outputs = []
            img_attns = []
            ques_attns = []

            for t in range(1, captions.size(1)):
                img_ctx, alpha_img = model.image_attention(h_t, image_feats)
                q_ctx, alpha_q = model.question_attention(h_t, question_outputs)

                logits, h_t, c_t = model.decoder.forward_step(input_token, h_t, c_t, img_ctx, q_ctx)
                outputs.append(logits.unsqueeze(1))

                img_attns.append(alpha_img)
                ques_attns.append(alpha_q)

                input_token = captions[:, t]

            outputs = torch.cat(outputs, dim=1)
            loss, ce, attn = compute_loss(outputs, captions, img_attns, ques_attns,
                                          lambda_img=lambda_img, lambda_ques=lambda_ques)

            total_loss += loss.item()
            total_ce += ce
            total_att += attn
            count += 1

            if save_outputs:
                pred_ids = outputs.argmax(dim=-1)  # [B, T]
                for pred, target in zip(pred_ids, captions[:, 1:]):
                    pred_tokens = tokenizer.decode(pred.tolist())
                    target_tokens = tokenizer.decode(target.tolist())
                    all_predictions.append(pred_tokens)
                    all_targets.append(target_tokens)

    avg_loss = total_loss / count
    avg_ce = total_ce / count
    avg_att = total_att / count

    print(f"[Eval] Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, Attn Reg: {avg_att:.4f})")

    if save_outputs:
        return avg_loss, all_predictions, all_targets
    else:
        return avg_loss

def get_max_seq_lengths(dataset):
    q_lens = []
    a_lens = []

    for sample in dataset:
        q_lens.append(len(sample['question_idxs']))
        a_lens.append(len(sample['caption_idxs']))

    q_lens = np.array(q_lens)
    a_lens = np.array(a_lens)

    print(" Question :")
    print(f"     max: {q_lens.max()}")
    print(f"     avg: {q_lens.mean():.2f}")
    print(f"  median: {np.median(q_lens)}")
    print(f"     std: {q_lens.std():.2f}")
    print(f"   80pct: {np.percentile(q_lens, 80)}")

    print("\n Answer :")
    print(f"     max: {a_lens.max()}")
    print(f"     avg: {a_lens.mean():.2f}")
    print(f"  median: {np.median(a_lens)}")
    print(f"     std: {a_lens.std():.2f}")
    print(f"   80pct: {np.percentile(a_lens, 80)}")

    return q_lens.max(), a_lens.max()

def get_seqs(model, tokenizer, image_pth, question = "Describe what's in the picture"):
    model.eval()
    transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    image = transform(Image.open(image_pth)).unsqueeze(0).to(model.device)
    
    question_idxs, _ = tokenizer.encode(question)
    question_tensor = torch.tensor(question_idxs).unsqueeze(0).to(model.device)
    with torch.no_grad():
        gen_ids, image_attn_log, question_attn_log, gate_log = model.generate(
                                                        image, 
                                                        question_ids=question_tensor, 
                                                        question_lengths= (question_tensor != 0).sum(dim=1), 
                                                        max_len=128, 
                                                        bos_token_id=1, eos_token_id=2, 
                                                        return_log=True
                                                        )
    return  gen_ids, image_attn_log, question_attn_log, gate_log

def visualize_attention(image_path, alphas, caption, encoded_image_size=14, k=5):
    """
    Visualize Attention for each word in the caption.

    :param image_path: Path to the input image
    :param alphas: Attention weights of shape (seq_length, num_pixels)
    :param caption: Generated caption as a list of indices
    :param encoded_image_size: Size of the encoder's grid (e.g., 14 for 14x14)
    :param k: Maximum number of plots per row
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((448, 448))  # Resize
    img_np = np.array(img)
    # Reshape alphas to grid shape (seq_length, encoded_image_size, encoded_image_size)
    alphas = np.array(alphas)  # (seq_length, num_pixels)
    alphas = alphas.reshape(len(alphas), encoded_image_size, encoded_image_size)
    # Caption words
    caption_arr = caption.split(' ')
    # Calculate number of rows needed
    # Plot original image and attention maps
    num_words = len(caption_arr)
    num_rows = int(np.ceil(num_words / k))
    plt.figure(figsize=(k * 5, num_rows * 5))
    for t, word in enumerate(caption_arr):
        if t >= len(alphas):  # Avoid plotting extra words
            break
        # Determine the position in the subplot grid
        plt.subplot(num_rows, k, t + 1)
        # Attention heatmap
        alpha = alphas[t]
        alpha = cv2.resize(alpha, (448, 448))  # Resize
        # Handle normalization safely
        if alpha.max() == alpha.min():
            alpha = np.zeros_like(alpha)  # No attention difference
        else:
            alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min() + 1e-9)  # Normalize to [0, 1]
        heatmap = cv2.applyColorMap((255 - (alpha * 255)).astype(np.uint8), cv2.COLORMAP_JET)
        # Overlay heatmap on the image
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        # Display the overlay with the word
        plt.imshow(overlay)
        plt.title(word, fontsize=28)
        plt.axis('off')
    plt.tight_layout(pad=1)
    plt.show()
    return

def parse_args():
    parser = argparse.ArgumentParser(description="Train DAMM Model")
    parser.add_argument('--json_path', type=str, default='./data/chat/llava_instruct_qa_pairs.json', help="Path to LLaVA chat.json file")
    parser.add_argument('--image_dir', type=str, default='./data/images', help="Path to LLaVA image folder")
    parser.add_argument('--vocab_path', type=str, default='./data/vocabulary.npz', help="Path to vocabulary")
    parser.add_argument('--save_path', type=str, default='./checkpoints', help="Path to save checkpoint files")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to checkpoint file")
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--decoder_hidden_dim', type=int, default=768)
    parser.add_argument('--question_hidden_dim', type=int, default=768)
    parser.add_argument('--image_feat_dim', type=int, default=2048)
    parser.add_argument('--attention_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--lambda_img', type=float, default=1.0)
    parser.add_argument('--lambda_ques', type=float, default=1.0)
    parser.add_argument('--min_freq', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    return args

if  __name__ == '__main__':
    args = parse_args()
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("!nltk.data.find")
        nltk.download('punkt_tab')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        raise RuntimeError("NLTK tokenizer data not found. Please download 'punkt_tab' using nltk.download('punkt_tab').")
    print("Loading Vocabulary...")
    vocab = Vocabulary()
    
    if not os.path.exists(args.vocab_path):
        if not os.path.exists(args.json_path):
            raise FileNotFoundError(f"[ERROR] chat.json data not found: {args.json_path}\n"
                                    f"Please check the path or ensure the data is generated.")
        try:
            vocab.build_from_chat_json(args.json_path, min_freq=args.min_freq)
            vocab.save(args.vocab_path)
            print(f"[INFO] Vocabulary built from {args.json_path} and saved to {args.vocab_path}")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to build vocabulary from {args.json_path}: {e}")
    else:
        print(f"[INFO] Using existing vocabulary: {args.vocab_path}")
        vocab.load(args.vocab_path)
    vocab_size=len(vocab)

    print("[INFO] Loading train_dataset...")
    train_dataset = LLaVAQADataset(
        json_path=args.json_path,
        image_dir=args.image_dir,
        vocab=vocab
    )
    train_dataloader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers= (os.cpu_count()//2),
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=8)
    
    print("[INFO] Train...")
    train_main(
        train_dataloader=train_dataloader,
        tokenizer=vocab,
        vocab_size=vocab_size,
        config={
            'embedding_dim': args.embedding_dim,
            'decoder_hidden_dim': args.decoder_hidden_dim,
            'question_hidden_dim': args.question_hidden_dim,
            'image_feat_dim': args.image_feat_dim,
            'attention_dim': args.attention_dim,
            'dropout': args.dropout,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'lambda_img': args.lambda_img,
            'lambda_ques': args.lambda_ques,
            'save_path': args.save_path
        },
        pre_ckpt=args.checkpoint_path
    )