# Tri de déchets par segmentation sémantique
**Binôme : Habib, Thibaud**

## Architecture du projet

```
waste_segmentation/
├── config.py             # Tous les hyperparamètres et chemins
├── mask_generation.py    # Génération automatique des masques (fond blanc → seuillage HSV)
├── scene_composer.py     # Composition de scènes synthétiques à densité variable
├── dataset.py            # PyTorch Dataset + DataLoaders + augmentations
├── model.py              # U-Net ResNet-50 + CombinedLoss (CE + Dice)
├── train.py              # Boucle d'entraînement (warmup cosine, early stopping)
├── evaluate.py           # mIoU, F1, confusion, analyse densité
├── main.py               # Point d'entrée (--step all/masks/scenes/train/eval)
└── requirements.txt
```

## Classes de déchets

| Index | Classe     | Couleur overlay |
|-------|------------|-----------------|
| 0     | background | noir            |
| 1     | glass      | cyan            |
| 2     | cardboard  | brun            |
| 3     | paper      | blanc           |
| 4     | plastic    | rouge           |
| 5     | metal      | gris            |
| 6     | trash      | vert            |

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Obtenir les données

```bash
python main.py --step download
```
Suivre les instructions affichées pour télécharger TrashNet et/ou le dataset Kaggle.

Placer les images dans :
```
data/raw/
  glass/       *.jpg
  cardboard/   *.jpg
  paper/       *.jpg
  plastic/     *.jpg
  metal/       *.jpg
  trash/       *.jpg
```

### 2. Pipeline complet

```bash
python main.py --step all
```

Ou étape par étape :

```bash
python main.py --step masks    # Génération des masques (seuillage HSV)
python main.py --step scenes   # Scènes synthétiques (sparse/medium/dense)
python main.py --step train    # Entraînement U-Net ResNet-50
python main.py --step eval     # Évaluation et analyse densité
```

## Méthodologie

### Génération de masques
Les images de déchets sont sur fond blanc. On convertit en HSV et on seuille :
- Saturation < 30 **ET** Valeur > 240 → pixel **fond**
- Sinon → pixel **objet** (étiqueté avec l'indice de sa classe)

Nettoyage par opérations morphologiques + sélection du plus grand composant connexe.

### Scènes synthétiques (plus utile ici)
On compose plusieurs déchets sur un canvas blanc en faisant varier le nombre d'objets :

| Densité | Nb objets | Chevauchement autorisé |
|---------|-----------|------------------------|
| sparse  | 3         | 5 %                    |
| medium  | 7         | 30 %                   |
| dense   | 14        | 60 %                   |

### Modèle
**U-Net avec encodeur ResNet-50 pré-entraîné** (ImageNet).

- Encodeur : ResNet-50 gelé (fine-tunable)
- Décodeur : 5 blocs `ConvTranspose2d + skip connection + 2×ConvBnReLU`
- Perte : `0.5 × CrossEntropy pondérée + 0.5 × Dice Loss`
- Optimiseur : AdamW + Warmup cosinus
- Input/Output : `(B, 3, 512, 512)` → `(B, 7, 512, 512)`

### Métriques
- **mIoU** (principale) — mean Intersection over Union
- Pixel Accuracy
- F1 par classe
- Dice coefficient
- Matrices de confusion par niveau de densité

## Sorties

Tout est sauvegardé dans `outputs/` :

| Fichier | Description |
|---------|-------------|
| `training_curves.png` | Courbes loss / mIoU |
| `density_comparison.png` | mIoU, acc, Dice par densité |
| `per_class_iou.png` | IoU par classe × densité |
| `confusion_sparse.png` | Matrice de confusion sparse |
| `confusion_medium.png` | Matrice de confusion medium |
| `confusion_dense.png` | Matrice de confusion dense |
| `density_analysis.csv` | Tableau complet des métriques |
| `training_log.csv` | Log epoch par epoch |
| `pred_vis_*.png` | Visualisations qualitatives |
