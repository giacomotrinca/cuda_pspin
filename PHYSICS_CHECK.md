# Physics Verification Report

Confronto sistematico tra la fisica dell'articolo di riferimento
(**Antenucci, Crisanti, Leuzzi, *Phys. Rev. Lett.* 114, 043901 (2015)**)
e l'implementazione nel codice CUDA.

---

## 1. Hamiltoniana

### Articolo (ACL 2015)

L'Hamiltoniana del modello p-spin complesso per il random laser è:

$$
H = H_2 + H_4
$$

con:

$$
H_2 = -\sum_{i<j} \mathrm{Re}\!\left[ J^{(2)}_{ij}\, a_i\, a_j^* \right]
$$

$$
H_4 = -\sum_{i<j<k<l} \sum_{\mathrm{ch}} \mathrm{Re}\!\left[ J^{(4)}_{ijkl,\mathrm{ch}}\; \prod_{\mu=0}^{3} f_\mathrm{ch}(a_{s_\mu}) \right]
$$

dove $a_i \in \mathbb{C}$ sono le ampiezze dei modi, e le tre coniugazioni
corrispondono ai tre modi di partizionare quattro indici in due coniugati
e due non-coniugati.

### Codice

| Elemento | Articolo | Codice | Stato |
|----------|----------|--------|-------|
| $H_2 = -\sum_{i<j}\mathrm{Re}[g_{ij} a_i a_j^*]$ | ✓ | [hamiltonian.h](include/hamiltonian.h): riga 7, [hamiltonian.cu](src/hamiltonian.cu): `energy_h2_kernel` | ✅ Conforme |
| $H_4$: 3 canali di coniugazione | ✓ | [disorder.h](include/disorder.h): righe 9–13, [hamiltonian.cu](src/hamiltonian.cu): `energy_h4_kernel` righe 66–113 | ✅ Conforme |
| Pattern di coniugazione: ch0={k,l}, ch1={j,k}, ch2={j,l} | ✓ | Codice: mask 0xC, 0x6, 0xA in [mc.cu](src/mc.cu) | ✅ Conforme |
| Somma su $i<j<k<l$ (combinazioni ordinate) | ✓ | Indice combinatoriale con decodifica sequenziale | ✅ Conforme |

**Risultato**: L'Hamiltoniana è implementata correttamente in entrambe le varianti (dense e sparse).

---

## 2. Vincolo e spazio delle fasi

### Articolo

Il modello sferico impone il vincolo:

$$
\sum_{i=1}^{N} |a_i|^2 = N
$$

(ipersfera $S^{2N-1}$ di raggio $\sqrt{N}$ in $\mathbb{R}^{2N}$).

### Codice

| Variante | Vincolo | MC move | File | Stato |
|----------|---------|---------|------|-------|
| **Spherical** (standard) | $\sum_i \|a_i\|^2 = N$ | Conserva $\|a_i\|^2 + \|a_j\|^2$ via rotazione + ridistribuzione modulo | [spins.h](include/spins.h), [spins.cu](src/spins.cu) | ✅ Conforme all'articolo |
| **Smoothed cube** (nuovo) | $\sum_i \|a_i\|^4 = N$ | Conserva $\|a_i\|^4 + \|a_j\|^4$ via angolo $\alpha$ | [spins_sparse.h](include/spins_sparse.h), [spins_sparse.cu](src/spins_sparse.cu) | ⚠️ Estensione (non nell'articolo) |

### Dettaglio: mossa MC sferica

```
a_i' = a_i * e^{-i*phi1} * sqrt((r1+r2)/r1) * cos(alpha)
a_j' = a_j * e^{-i*phi2} * sqrt((r1+r2)/r2) * sin(alpha)
```

dove $r_k = |a_k|^2$. Verifica:

$$
|a_i'|^2 + |a_j'|^2 = (r_1+r_2)\cos^2\alpha + (r_1+r_2)\sin^2\alpha = r_1+r_2 \quad \checkmark
$$

**Conservazione globale**: poiché ogni mossa modifica solo due spin e conserva la somma delle loro norme quadre, il vincolo globale $\sum_i |a_i|^2 = N$ è preservato esattamente ad ogni step. ✅

### Dettaglio: mossa MC smoothed cube

```
|a_i'|^2 = sqrt(S4) * cos(alpha)
|a_j'|^2 = sqrt(S4) * sin(alpha)
```

con $S_4 = |a_i|^4 + |a_j|^4$. Verifica:

$$
|a_i'|^4 + |a_j'|^4 = S_4\cos^2\alpha + S_4\sin^2\alpha = S_4 \quad \checkmark
$$

**Conservazione globale**: il vincolo $\sum_i |a_i|^4 = N$ è preservato. ✅

---

## 3. Disordine e accoppiamenti

### Articolo

- Accoppiamenti i.i.d. gaussiani complessi (media nulla)
- Varianza scalata per l'estensività: $\mathrm{Var}(J) \propto N / n_\text{terms}$

### Codice

| Elemento | Codice | Stato |
|----------|--------|-------|
| $g_2$ gaussiani reali, simmetrici ($g_{ij}=g_{ji}$), diagonale zero | [disorder.cu](src/disorder.cu): `generate_g2` + `zero_imag_kernel` + `symmetrize_g2_kernel` | ✅ Conforme |
| $g_4$ tensore simmetrico: un solo coupling per quartetto, replicato ai 3 canali | [disorder.cu](src/disorder.cu): `generate_g4` → genera $\binom{N}{4}$ + `cudaMemcpy` Device→Device | ✅ Conforme |
| FMC indipendente per canale (stessi coupling, diversa condizione) | [disorder.cu](src/disorder.cu): `fmc_filter_g4_kernel` azzera canali indipendentemente | ✅ Conforme |
| Rescaling: $\sigma = J\sqrt{N/n_\text{surviving}}$ | [disorder.cu](src/disorder.cu): `rescale_g2`, `rescale_g4` | ✅ Conforme |

**Nota sugli accoppiamenti reali**: il codice genera accoppiamenti gaussiani **reali** (azzera la parte immaginaria con `zero_imag_kernel`). Nell'articolo ACL2015 gli accoppiamenti $J_{ij}$ sono definiti come gaussiani (la fase non è specificata esplicitamente nel modello mean-field). La scelta di accoppiamenti reali è una specializzazione compatibile con il framework generale — non altera la classe di universalità, ma riduce le fluttuazioni di fase. **Accettabile** per il modello studiato.

**Simmetria $g_4$**: fisicamente il tensore di accoppiamento $g_{ijkl}$ proviene dall'integrale di sovrapposizione spaziale dei profili di modo, che è totalmente simmetrico sotto permutazioni degli indici. Il codice ora genera un solo coupling per quartetto e lo replica ai 3 canali di coniugazione (ch 0, 1, 2). Il filtro FMC agisce poi indipendentemente su ciascun canale. ✅

**Simmetria $g_2$**: il `symmetrize_g2_kernel` copia esplicitamente il triangolo superiore in quello inferiore e azzera la diagonale (nessuna auto-interazione). Funzionalmente cosmetico: il codice accede solo al triangolo superiore ($i < j$), ma garantisce consistenza logica del tensore. ✅

---

## 4. Frequency Matching Condition (FMC)

### Articolo

Il filtro FMC modella la quasi-risonanza nelle cavità laser. Per i termini a 4 corpi, la condizione è:

$$
|\omega_i + \omega_j - \omega_k - \omega_l| \leq \gamma
$$

dove la ripartizione tra frequenze "positive" e "negative" segue il canale di coniugazione.

### Codice

| FMC Mode | H2 filter | H4 filter | File | Stato |
|----------|-----------|-----------|------|-------|
| FC (`fmc=0`) | Nessuno | Nessuno | — | ✅ Caso fully-connected |
| Comb (`fmc=1`) | H2 completamente ucciso (nessuna coppia sopravvive) | $\|\omega_i+\omega_j-\omega_k-\omega_l\| \leq \gamma$ con $\omega_k=k$ | [disorder.cu](src/disorder.cu) | ✅ Conforme |
| Uniform (`fmc=2`) | $\|\omega_i-\omega_j\| \leq \gamma$ | Come comb | [disorder.cu](src/disorder.cu) | ✅ Conforme |

**Condizioni per canale** (codice, [disorder.cu](src/disorder.cu) righe 137–145):

| Canale | Condizione | Coniugati |
|--------|-----------|-----------|
| ch 0 | $\|\omega_{ii}+\omega_{jj} - \omega_{kk}-\omega_{ll}\| \leq \gamma$ | $\{k,l\}$ |
| ch 1 | $\|\omega_{ii}+\omega_{ll} - \omega_{jj}-\omega_{kk}\| \leq \gamma$ | $\{j,k\}$ |
| ch 2 | $\|\omega_{ii}+\omega_{kk} - \omega_{jj}-\omega_{ll}\| \leq \gamma$ | $\{j,l\}$ |

Queste sono coerenti con il pattern di coniugazione: le frequenze "positive" corrispondono ai modi non-coniugati, quelle "negative" ai coniugati. ✅

---

## 5. Mossa Monte Carlo e Metropolis

### Articolo

Campionamento canonico a temperatura $T = 1/\beta$ con criterio di Metropolis:

$$
P(\text{accept}) = \min\!\left(1,\; e^{-\beta\,\Delta E}\right)
$$

### Codice

[mc.cu](src/mc.cu) righe 296–310:

```cuda
bool accept = (dE <= 0.0);
if (!accept) {
    double r = curand_uniform_double(&rng);
    accept = (r < exp(-beta * dE));
}
```

| Aspetto | Stato |
|---------|-------|
| Criterio Metropolis standard | ✅ |
| $T = \infty$ ($\beta = 0$): accetta tutto | ✅ |
| $N/2$ passi per sweep (ogni passo aggiorna 2 spin) | ✅ Ergodico |

---

## 6. Calcolo di $\Delta E$

### H2: fattorizzazione

Il codice usa una fattorizzazione intelligente per $H_2$. Dato che:

$$
\Delta H_2 = -\mathrm{Re}\!\left[\Delta a_i \sum_{k\neq i,j} g_{ik} a_k^* + \Delta a_j \sum_{k\neq i,j} g_{jk} a_k^*\right] - \Delta(\text{coppia} \; i,j)
$$

questo è implementato in [mc.cu](src/mc.cu) come somma separata sui contributi di $i_0$ e $j_0$, con il termine $(i_0, j_0)$ aggiunto separatamente dal thread 0. ✅

### H4: enumerazione a tre tipi (dense)

Per la variante densa, il calcolo di $\Delta H_4$ enumera solo i quartetti che contengono $i_0$ e/o $j_0$:

- **Type 1**: $\{i_0, a, b, c\}$ — solo $i_0$ cambia → update differenziale
- **Type 2**: $\{j_0, a, b, c\}$ — solo $j_0$ cambia → update differenziale
- **Type 3**: $\{i_0, j_0, a, b\}$ — entrambi cambiano → ricalcolo completo

Il conteggio:
- Type 1 & 2: $\binom{N-2}{3}$ ciascuno
- Type 3: $\binom{N-2}{2}$
- Totale: $2\binom{N-2}{3} + \binom{N-2}{2}$ — corretto, copre tutti i quartetti che contengono $i_0$ o $j_0$ o entrambi. ✅

---

## 7. Inizializzazione degli spin

### Spherical

[spins.cu](src/spins.cu): genera $2N$ numeri gaussiani i.i.d. (parte reale + immaginaria), poi proietta sulla sfera $\sum|a_i|^2 = N$. Questo dà una distribuzione uniforme su $S^{2N-1}$. ✅

### Smoothed cube

[spins_sparse.cu](src/spins_sparse.cu): genera $2N$ gaussiani, poi riscala di un fattore $(N/\sum|a_i|^4)^{1/4}$ per proiettare sulla superficie $\sum|a_i|^4 = N$.

⚠️ **Nota**: questa proiezione radiale NON genera una distribuzione uniforme sulla superficie $\sum|a_i|^4 = N$, ma dato che il campionamento MC poi raggiunge la distribuzione stazionaria, l'inizializzazione è solo un punto di partenza e non influenza i risultati a regime.

---

## 8. Parallel Tempering

Il parallel tempering è standard: ad ogni iterazione si propongono scambi tra repliche adiacenti in temperatura con probabilità:

$$
P(\text{swap}) = \min\!\left(1,\; e^{(\beta_a - \beta_b)(E_a - E_b)}\right)
$$

Non ho verificato il codice dello swap in dettaglio (è in `parallel_tempering.cu`), ma la struttura è quella canonica.

---

## 9. Variante Sparse H4

Nella variante sparse ([mc_sparse.cu](src/mc_sparse.cu)), dopo il filtro FMC, vengono campionati solo $N$ quartetti dall'insieme dei sopravviventi (Fisher-Yates parziale). I couplings vengono riscalati di conseguenza.

Questo è un'**approssimazione** che riduce la complessità da $O(N^3)$ a $O(N)$ per step MC. Non è presente nell'articolo di riferimento, ma è una scelta computazionale ragionevole per esplorare taglie grandi.

---

## 10. Riepilogo delle differenze rispetto all'articolo

| # | Differenza | Impatto | Azione suggerita |
|---|-----------|---------|------------------|
| 1 | **Smoothed cube** constraint ($\sum\|a\|^4 = N$) | Nuovo vincolo geometrico, diversa classe universale potenziale | Documentare bene. Studio sistematico delle differenze vs sferico |
| 2 | **Sparse H4** (N quartetti su ~$N^3$ sopravviventi) | Approssimazione: riduce l'interazione effettiva | Verificare convergenza dell'energia media vs versione densa |
| 3 | **Accoppiamenti reali** (non complessi) | Reduce simmetria da U(1) → Z₂ sugli accoppiamenti | Minimo per modello mean-field; commentare nel codice |
| 4 | **Init cube non uniforme** | Nessuno (MC termalizza comunque) | Nessuna |
| ~~5~~ | ~~**g4 indipendenti per canale**~~ | ~~Corretto~~ | ✅ **RISOLTO**: coupling replicato da singolo quartetto |
| ~~6~~ | ~~**g2 non simmetrizzato**~~ | ~~Cosmetico~~ | ✅ **RISOLTO**: `symmetrize_g2_kernel` aggiunto |

---

## 11. Check list di consistenza interna del codice

| Check | Risultato |
|-------|-----------|
| Mossa MC conserva il vincolo (sferico/cube) | ✅ Verificato analiticamente |
| $\Delta E$ corretto per H2 (fattorizzazione) | ✅ |
| $\Delta E$ corretto per H4 (3 tipi, 3 canali) | ✅ |
| Metropolis implementato correttamente | ✅ |
| FMC: condizione di risonanza per canale | ✅ |
| Rescaling disordine: $\sigma = J\sqrt{N/n_\text{surv}}$ | ✅ |
| Riduzione warp-shuffle per $\Delta E$ | ✅ |
| Shared memory layout coerente | ✅ |

---

*Report generato automaticamente dall'analisi del codice sorgente e confronto con ACL2015.*
