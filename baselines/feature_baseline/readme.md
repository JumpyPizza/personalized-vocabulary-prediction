A feature-based classification baseline for **vocabulary knowledge prediction**.  Each learner labels a small set of words, and the model predicts their knowledge of the remaining words using **linguistic, psycholinguistic, and learner-level features**.

---

##  Inputs Required

### 1. Learner Annotations (`annotations_folder`)
- A folder where each learner is a file: `learner_id.txt`  
  Example:
0.txt
1.txt
...


- Each file contains tab-separated lines:
word<TAB>label

where `label` = `1` (known) or `0` (unknown).

---

### 2. Word Features 
Precomputed features for each word. Extracted from:

#### **Frequency Features (from COCA corpus)**
- Token frequency  
- Document frequency  
- Relative frequency (normalized by total tokens)  
- Band ID (frequency band index)

#### **Orthographic Features**
- Number of letters  
- Number of syllables (heuristic or via NLP library)  
- Number of phonemes (from MRC if available)  

#### **Psycholinguistic Features (from MRC norms)**
- Familiarity  
- Concreteness  
- Imageability  
- Age of Acquisition (AoA) rating  
- Meaningfulness (Colorado norms)  
- Meaningfulness (Paivio norms)  
- KF Written Frequency  
- KF Number of Categories  
- KF Number of Samples  
- Thorndike–Lorge Frequency  
- Brown Verbal Frequency  

#### **CEFR Difficulty**
- CEFR level (A1, A2, B1, B2, C1, C2, or UNK) from CEFR word list

---

### 3. Learner-Level Features (computed at training time)
- **learner_known_rate**: fraction of training words labeled as “known”  
- **learner_phi**: φ-weighted ability score, based on frequency bands