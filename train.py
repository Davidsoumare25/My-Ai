import os
import re
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. Configuration des fichiers
FICHIER_TEXTE = "test.txt"
FICHIER_CERVEAU = "cerveau_senegal.pt"
FICHIER_VOCAB = "vocab_senegal.txt"

if not os.path.exists(FICHIER_TEXTE):
    print(f"Erreur : Le fichier '{FICHIER_TEXTE}' est introuvable.")
    exit()

# Détection automatique du matériel (GPU de Colab ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Matériel utilisé pour l'entraînement : {device}")

# 2. Indexation et Tokenisation du Vocabulaire
print("Indexation du vocabulaire...")
vocab_set = {"<pad>", "<sos>", "<eos>", "<unk>"}
with open(FICHIER_TEXTE, "r", encoding="utf-8") as f:
    for line in f:
        mots = re.findall(r"\w+", line.lower())
        vocab_set.update(mots)

vocab = sorted(list(vocab_set))
vocab_size = len(vocab)
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}

# Sauvegarde du dictionnaire pour le script de discussion
with open(FICHIER_VOCAB, "w", encoding="utf-8") as f:
    f.write("\n".join(vocab))

print(f"Taille du vocabulaire national : {vocab_size} tokens.")

# 3. Hyperparamètres du Transformer
D_MODEL = 256         # Dimension de l'espace sémantique
N_HEADS = 4           # Nombre de têtes d'attention simultanées
N_LAYERS = 3          # Nombre de blocs Transformer empilés
D_FF = 512            # Dimension de la couche Feed-Forward interne
MAX_LEN = 30          # Longueur maximale d'une phrase
BATCH_SIZE = 64       # Nombre de phrases traitées en parallèle
LEARNING_RATE = 5e-4
EPOCHS = 5

# 4. Dataset PyTorch pour le chargement des données
class DialogueDataset(Dataset):
    def __init__(self, fichier, word_to_idx, max_len):
        self.sequences = []
        with open(fichier, "r", encoding="utf-8") as f:
            for line in f:
                mots = re.findall(r"\w+", line.lower())
                if len(mots) < 1: continue
                
                # Construction des indices avec jetons de début (sos) et de fin (eos)
                indices = [word_to_idx["<sos>"]] + [word_to_idx.get(m, word_to_idx["<unk>"]) for m in mots] + [word_to_idx["<eos>"]]
                
                if len(indices) < max_len:
                    indices += [word_to_idx["<pad>"]] * (max_len - len(indices))
                else:
                    indices = indices[:max_len]
                self.sequences.append(torch.tensor(indices, dtype=torch.long))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Pour l'entraînement causal (génération), l'entrée exclut le dernier token, la cible décale d'un token
        seq = self.sequences[idx]
        return seq[:-1], seq[1:]

dataset = DialogueDataset(FICHIER_TEXTE, word_to_idx, MAX_LEN + 1)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# 5. Architecture du Réseau Transformer (Décodeur Causal)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SenegalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=word_to_idx["<pad>"])
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Utilisation des blocs natifs PyTorch pour l'Attention Multi-Têtes
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, 
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        # Masque causal pour empêcher le modèle de regarder les mots futurs
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, x):
        sz = x.size(1)
        mask = self.generate_square_subsequent_mask(sz).to(x.device)
        
        out = self.embedding(x) * math.sqrt(D_MODEL)
        out = self.pos_encoder(out)
        # Dans un décodeur pur, la mémoire (contexte de l'encodeur) est remplacée par le vecteur lui-même
        out = self.transformer(tgt=out, memory=out, tgt_mask=mask, memory_mask=mask)
        return self.fc_out(out)

# Initialisation du modèle
model = SenegalTransformer(vocab_size, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_LEN).to(device)

# 6. Boucle d'Entraînement Industrielle
criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx["<pad>"])
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print("\n=== DÉMARRAGE DE L'ENTRAÎNEMENT DU TRANSFORMER ===")
model.train()

for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Reshape pour calculer l'erreur sur l'ensemble du batch aplati
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        
        # Gradient Clipping pour stabiliser l'apprentissage
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"Époque {epoch}/{EPOCHS} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
            
    scheduler.step()
    print(f"--- Fin de l'Époque {epoch} | Perte Moyenne: {total_loss / len(dataloader):.4f} ---")

# 7. Sauvegarde du modèle complet
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'd_model': D_MODEL,
    'n_heads': N_HEADS,
    'n_layers': N_LAYERS,
    'd_ff': D_FF,
    'max_len': MAX_LEN
}, FICHIER_CERVEAU)

print(f"\n[SUCCÈS] Le cerveau Transformer Local '{FICHIER_CERVEAU}' a été généré sur Colab !")

