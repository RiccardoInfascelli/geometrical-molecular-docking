# Prova con un ensemble di conformeri

Questa directory contiene una prova CPU su un solo complesso, separata dal
motore principale. Gli script generano conformeri esterni; il driver di
ricerca riusa preparazione, matching, Kabsch, clash e Fast del motore.
Un secondo driver ricalcola WeightedInteraction e Vinardo rep50 sulle trasformazioni salvate
e ordina le pose con **Hybrid globale, lambda=0.3 e alpha=0.5, senza Fast
prefilter**. Non serve una GPU nel Docker del Mac.

**L'ultima estensione Hybrid globale è work-in-progress: non è ancora
compilata e validata numericamente.** I comandi descrivono il flusso
implementato da verificare; non attestano un risultato di accuratezza.

## Ambiente

La generazione richiede Python RDKit con ETKDGv3 e MMFF. Le librerie C++ e il
modulo Python sono dipendenze distinte: la disponibilità va controllata
nell'interprete effettivamente usato, anche quando esiste un ambiente virtuale.
Il replay Hybrid di un pool già salvato usa Python standard e gli scorer C++:
non richiede Python RDKit e non rigenera conformeri o pose.

Dal Mac, prima della prova:

```sh
docker exec -w /workspace/geometric-docking geometric-docking-dev \
  python3 -B experiments/conformer_ensemble/prepare_conformers.py --check
```

Se il controllo fallisce, usare un ambiente RDKit già predisposto oppure
segnalare la dipendenza mancante, senza installarla automaticamente.
Il runner completo controlla RDKit prima di aprire molecole o creare output.

## Compilazione e test

I comandi seguenti si eseguono sul Mac. Creano una build separata CPU con le
dipendenze C++ già richieste dal progetto. La dipendenza attuale imposta
`CMP0167 NEW`: serve un CMake che supporti quella policy, 3.30 o successivo.
Non si modifica la dipendenza per questo esperimento.

```sh
docker exec -w /workspace/geometric-docking geometric-docking-dev \
  cmake -S experiments/conformer_ensemble \
        -B experiments/conformer_ensemble/build -DCMAKE_BUILD_TYPE=Release

docker exec -w /workspace/geometric-docking geometric-docking-dev \
  cmake --build experiments/conformer_ensemble/build \
        --target conformer_docking conformer_hybrid \
        conformer_hybrid_ranking_test --parallel 1

docker exec -w /workspace/geometric-docking geometric-docking-dev \
  ctest --test-dir experiments/conformer_ensemble/build --output-on-failure
```

La suite autonoma comprende test sintetici della preparazione, del replay e
del ranking globale, più un controllo esplicito di disponibilità RDKit.
Quest'ultimo fallisce se il modulo manca. Per eseguire soltanto i controlli
Python del pool salvato, senza RDKit:

```sh
docker exec -w /workspace/geometric-docking geometric-docking-dev \
  python3 -B experiments/conformer_ensemble/hybrid_test.py -v
```

`conformer_test.py` contiene anche un test sintetico con RDKit, saltato quando
il modulo manca. `hybrid_ranking_test.cpp` controlla normalizzazione comune,
Top-K locali, Top-X globale, parità, pool vuoti/costanti e indipendenza
dall'RMSD; `hybrid_test.py` controlla completezza e identità dei record salvati,
assenza di prefilter e protezione degli output. Sono verifiche predisposte,
non una validazione C++ già completata del nuovo percorso.

## Riutilizzare i conformeri e le pose di 1a1e

Se il pool completo esiste già, non ripetere ETKDG, MMFF o docking. Esempio
per il run già generato, dopo avere compilato e controllato il nuovo driver:

```sh
docker exec -w /workspace/geometric-docking geometric-docking-dev \
  python3 -B experiments/conformer_ensemble/rank_hybrid.py \
    --run experiments/conformer_ensemble/runs/1a1e-20261023T062143Z-433 \
    --complex /workspace/datasets/refined-set/1a1e \
    --local-k 10 --top-x 10
```

`--run` identifica la directory che contiene `preparation/`, `docking/` e
`protocol.json`. Il replay controlla che la ricerca sia conclusa, i record
siano completi e le trasformazioni rappresentino tutti i superstiti originali.
Il protocollo deve dichiarare assenza di prefilter e indice shared-only.
CSV troncati o ID duplicati producono errore: non si completa un pool mancante
scartando silenziosamente righe.

L'output predefinito è una directory nuova
`runs/1a1e-hybrid-<data>-<pid>/`, separata dal run sorgente.
`--output DIR` ne sceglie un'altra e `--driver FILE` seleziona una build diversa
di `conformer_hybrid`. Gli originali rimangono invariati. Se il percorso del
complesso salvato proviene da un altro ambiente, `--complex` lo sostituisce
senza riscrivere il protocollo originale.

## Prima prova su 1a1e

Per generare un nuovo pool, soltanto se quello salvato non basta e dopo avere
verificato il nuovo percorso Hybrid:

```sh
docker exec -w /workspace/geometric-docking geometric-docking-dev \
  python3 -B experiments/conformer_ensemble/run_experiment.py \
    --complex /workspace/datasets/refined-set/1a1e \
    --num-conformers 10 --num-candidates 30 --seed 20261023 \
    --min-rmsd 0.5 --energy-window 10 --max-iterations 500
```

Gli input sono `1a1e_protein.pdb`, `1a1e_pocket.pdb` e `1a1e_ligand.mol2`, letti
senza modificarli. L'output è una directory nuova sotto
`experiments/conformer_ensemble/runs/1a1e-<data>-<pid>/`. `--output DIR` ne sceglie
un'altra, che deve ancora non esistere. `--driver FILE` sceglie la build del
generatore e `--hybrid-driver FILE` quella del rescoring. Il runner richiede
entrambi e non compila automaticamente. `--generation-only` salva soltanto
il pool completo, da rianalizzare poi con `rank_hybrid.py`.

## Preparazione e selezione

Il MOL2 deve descrivere un ligando connesso, con ID atomici ordinati da 1 a N.
Il tool controlla ordine, coordinate e nomi atomici rispetto alla lettura RDKit.
Non ricostruisce la molecola da SMILES né introduce una nuova scelta di
protonazione o typing. La lettura usa `cleanupSubstructures=True`, come il
parser MOL2 del motore, preservando i campi originali nel file esportato.
La sanificazione Python non viene dichiarata identica a quella personalizzata
del motore: il driver C++ verifica nuovamente il mapping dopo la propria lettura.

ETKDGv3 genera fino a 30 candidati, con seed fissato, un thread e controllo
della chiralità. Il pruning interno è disattivato: la diversità viene misurata
dopo la minimizzazione. MMFF94 è applicato se supporta tutti gli atomi, per
massimo 500 iterazioni. Energie non finite e conformeri non convergenti sono
esclusi. Se non rimane alcun candidato, la prova fallisce; non passa a UFF.

I candidati sono esaminati per energia crescente entro 10 kcal/mol dal minimo.
La selezione greedy conserva al massimo 10 conformeri distanti almeno 0,5 Å
da quelli già scelti. Usa RMSD heavy-atom allineato, mapping fisso, senza
riflessioni o correzione delle simmetrie. Sono soglie esplorative registrate,
non valori già validati su 1a1e. Possono rimanere meno di 10 conformeri:
non si allentano i filtri per raggiungere artificialmente il numero richiesto.

Se MMFF94 non supporta il ligando, restano i conformeri ETKDG e il solo filtro
di diversità, nell'ordine di generazione. Le energie sono assenti, non zero,
e il protocollo segnala l'assenza di selezione energetica.

Eventuali H aggiunti alla copia di lavoro rimangono in coda. L'export conserva
le prime N coordinate, senza `RemoveHs`, preservando anche tutti gli H originali.
`AddHs` non ripara automaticamente MOL2 carenti di idrogeni: questa prima
versione non è una procedura generale di preparazione della protonazione.

L'export cambia soltanto nome del record e coordinate ATOM: conserva ID, ordine,
nomi, tipi, cariche e legami. Ogni conformero viene riletto e confrontato con
grafo e stereochimica della sorgente immutabile. Le modifiche interne di
aromaticità di MMFF non vengono trasferite al MOL2. La baseline mantiene le
stringhe numeriche originali e non viene minimizzata.

## Docking e significato del confronto

`conformers.mol2` contiene prima la baseline, poi i conformeri selezionati come
record distinti. Il driver CPU prepara una sola volta proteina, griglia e indice
shared con cutoff 24 Å, senza pruning o fallback al diametro. Ogni record è
elaborato separatamente sullo stesso indice. Il runner sceglie quaternion e
clash cached, con profiling disattivato.

Le funzioni pubbliche generano tutti i superstiti e calcolano Fast con i pesi
predefiniti. La Top-1 Fast del generatore rimane una diagnostica; non limita
le pose esportate né entra nel ranking finale Hybrid. Il rescoring riusa
`score_weighted_interactions()` e `VinardoLikeScorer::score()` su tutte le pose
salvate. WeightedInteraction considera per default le aree della pocket; il replay ammette
`--interaction-region protein`. Vinardo usa la proteina completa supportata e lo
stesso cutoff atomico stretto di 8 Å, senza crop.

Il riferimento rimane il ligando cristallografico nel sistema della proteina,
anche per gli input ETKDG. Il mapping atomico completo, compresi H e grafo,
viene verificato. L'RMSD di docking è diretto, senza riallineamento né correzione
delle simmetrie; il successo è `RMSD < 2 Å`. È diverso dall'RMSD allineato
usato soltanto per selezionare conformeri geometricamente diversi.

Il ranker esistente calcola `V_rep50 = V + (0.5-1)*repulsion` e
`H = 0.3*z(WeightedInteraction) + 0.7*z(-V_rep50)`. Media e deviazione standard di
popolazione vengono calcolate **una sola volta sull'unione di tutti i
superstiti dei conformeri generati**, prima di qualunque Top-K. Un termine
costante ha z-score zero. La baseline cristallografica è esclusa da questo
pool e viene ordinata separatamente con la propria normalizzazione.

Con gli stessi score comuni si assegnano i ranghi locali e si conservano
le Top-10 di ciascun conformero. La loro unione produce la Top-10 globale
senza rinormalizzare. Per `K >= X` questa selezione conserva la Top-X del
pool completo: una posa oltre K nel suo conformero ha già almeno K pose
davanti. Le parità seguono record e `pose_id` originali crescenti, mai RMSD,
energia MMFF o Fast. Gli score Hybrid normalizzati della baseline non sono
valori assoluti da confrontare con quelli dell'ensemble.

## Output e tempi

| File | Contenuto |
| --- | --- |
| `preparation/conformers.mol2` | Baseline e conformeri selezionati, ordinati come il CSV. |
| `preparation/reference.mol2` | Riferimento derivato; l'originale resta intatto. |
| `preparation/conformers.csv` | ID, energia MMFF se disponibile, convergenza e diversità. |
| `preparation/candidates.csv` | Tutti i candidati e motivi di selezione/esclusione. |
| `preparation/protocol.json` | Versioni, seed, parametri, H, numerosità e tempi. |
| `docking/poses.csv` | Tutti i superstiti con Fast, RMSD e trasformazioni R/t a 17 cifre. |
| `docking/records.csv` | Conteggi, Top-1 Fast diagnostica, miglior RMSD disponibile e tempi per conformero. |
| `docking/protocol.csv` | Parametri del motore, pesi, indice e costo iniziale comune. |
| `fast_summary.csv`, `fast_comparison.csv` | Diagnostiche Fast del nuovo runner; non sono il risultato finale Hybrid. |
| `hybrid/ranking/global_rankings.csv` | Top-X globale: conformero, ID posa, ranghi locale/globale, WeightedInteraction, rep50, Hybrid, RMSD e successo. |
| `hybrid/ranking/local_rankings.csv` | Top-K locali con score normalizzati sul pool comune. |
| `hybrid/ranking/baseline_rankings.csv` | Baseline cristallografica classificata separatamente. |
| `hybrid/ranking/all_pose_scores.csv` | Tutte le pose, componenti Vinardo e score, con popolazione ensemble/baseline distinta. |
| `hybrid/ranking/topk_summary.csv` | Top-1/5/10, successo e migliore RMSD disponibile per le due popolazioni. |
| `hybrid/ranking/protocol.csv` | Regioni, formule, pool e tempi separati del rescoring. |
| `protocol.json` e log | Comando, ambiente, hash del driver e sorgenti sperimentali, exit status e wall time. |

Nel replay autonomo i file di ranking sono sotto `ranking/` della nuova
directory di analisi, non nel run sorgente. I run precedenti possono ancora
contenere `summary.csv` e `comparison.csv`: sono risultati Fast storici da
lasciare intatti, non export Hybrid da rinominare.

La Top-1 Hybrid è la posa effettivamente scelta. Il miglior RMSD nei primi
5 o 10 misura il successo Top-k; il minimo sull'intero pool è invece un
oracle: descrive cosa ha trovato la ricerca, non una selezione realizzabile
senza conoscere il riferimento. Il successo Top-k richiede almeno una posa
con `RMSD < 2 Å` nel prefisso ordinato. Nessuna di queste metriche modifica
lo score o la classifica.

`no_poses` e `no_triplets` hanno successo zero e RMSD vuoto. Un errore produce
exit status non nullo e protocollo fallito. I record non completati sono
`not_processed` e i confronti del run parziale sono marcati incompleti.
Non si trasformano errori o valori assenti in RMSD zero.

I tempi del generatore separano preparazione ligando, ricerca, Fast e RMSD.
`online_s` somma i primi tre, escludendo RMSD e output. Preparazione proteica e
indice sono un costo comune conteggiato una volta. Il wall del processo include
anche questi costi, RMSD ed export; quello complessivo aggiunge ETKDG/MMFF
e, nel nuovo runner completo, il rescoring Hybrid. Il replay registra
preparazione scorer, validazione input/RMSD, WeightedInteraction, Vinardo e ranking
separatamente; non è una nuova misura della ricerca geometrica.
Sono misure descrittive singole, non benchmark ripetuti. Non si deve omettere
il costo dei conformeri dal confronto end-to-end con una sola ricerca.

L'energia MMFF94 del ligando isolato non viene sommata agli score e non sceglie
la posa dopo il docking. Il minimo isolato non è assunto come conformazione
corretta nel sito. La baseline parte dalla conformazione legata e dalla pocket
nota; una prova su 1a1e non dimostra generalità né docking cieco.

## Riferimenti RDKit

API ufficiali: [ETKDG](https://www.rdkit.org/docs/RDKit_Book.html#conformer-generation),
[MMFF](https://www.rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html),
[MOL2](https://www.rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html),
[allineamento non mutante](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolAlign.html).
L'avvertenza sulle modifiche di aromaticità MMFF è nella
[guida C++](https://www.rdkit.org/docs/GettingStartedInC%2B%2B.html).
