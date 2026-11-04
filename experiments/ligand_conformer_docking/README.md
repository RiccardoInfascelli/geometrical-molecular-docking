# Docking CUDA di un ensemble di conformeri

Suite C++ autonoma in `experiments/ligand_conformer_docking/`.
Genera conformeri RDKit ed esegue docking rigido CUDA con le API del motore.
È separata dal workflow Python in `experiments/conformer_ensemble/`;
i ranking e le formule di scoring restano quelli del motore.

L'[esperimento CPU di qualità dei conformeri](README-conformer-quality.md)
valuta una sola sequenza ETKDGv3 con budget a prefisso, senza docking o MMFF.
Ha una policy separata che esclude coordinate e stereochimica native dalla
generazione; non usa il wrapper CUDA 1/5/10/20 descritto sotto.

## Compilazione e controlli host

Il generatore usa RDKit C++, senza Python RDKit o subprocessi per ETKDG/MMFF.
Servono le dipendenze C++ già previste dal progetto e i componenti RDKit di
embedding, force field e allineamento.

```sh
cmake -S experiments/ligand_conformer_docking -B build/conformer-host \
  -DENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DTARGET_CPU_ARCHITECTURE=generic
cmake --build build/conformer-host \
  --target ligand_conformer_docking conformer_generation_test ensemble_ranking_test \
    rmsd_oracle_test conformer_export_test ensemble_diagnostics_test --parallel 1
ctest --test-dir build/conformer-host \
  -R 'conformer_generation|ensemble_ranking|ensemble_diagnostics|rmsd_oracle|conformer_export' --output-on-failure
```

La build host supporta `--generate-only` e i test sintetici, non la ricerca CUDA.
`conformer_export` usa pose sintetiche e i writer reali per confrontare modalità
e dettagli di export, mantenendo invariati RMSD, oracoli e riepiloghi completi.
Per compilare e validare successivamente sulla A100, con le dipendenze configurate:

```sh
cmake -S experiments/ligand_conformer_docking -B build/conformer-cuda \
  -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DTARGET_CPU_ARCHITECTURE=generic \
  -DENABLE_OPENMP=ON -DDOCKING_TEST_COMPLEX=/percorso/1a1e
cmake --build build/conformer-cuda \
  --target ligand_conformer_docking conformer_original_test conformer_execution_test --parallel 1
ctest --test-dir build/conformer-cuda \
  -R '^conformer_(original|execution)$' --output-on-failure
```

`conformer_original` e `conformer_execution` richiedono CUDA e un complesso reale
configurato con `DOCKING_TEST_COMPLEX`. Il secondo confronta batch e overlap
con il percorso sequenziale di riferimento. Il wrapper Python non esegue i test automaticamente.
OpenMP è opzionale: senza di esso usare un thread per la preparazione delle query.

## Una configurazione

Proteina e pocket devono condividere il sistema di coordinate. Il riferimento
cristallografico resta in quel sistema; i conformeri iniziali possono essere
altrove. Usare una directory di output nuova:

```sh
build/conformer-cuda/ligand_conformer_docking /percorso/1a1e /percorso/nuovo-output \
  --num-conformers 10 --seed 20261027 --min-rmsd 0.5 \
  --fast-prefilter none --crop-margin off --cutoff 24
```

`--ligand MOL2` e `--reference MOL2` indicano input alternativi espliciti.
`--original-only` esegue la sola baseline senza ETKDG; `--generate-only`
esegue la generazione host senza ricerca CUDA. Quaternion Kabsch è imposto
dal processo: non esistono un selettore SVD o un'opzione `--kabsch`.

ETKDG usa il seed registrato; `--min-rmsd` controlla la diversità dei conformeri.
MMFF è disattivato per default e si attiva con `--mmff`; il limite è
`--max-iterations 500`. Se mancano parametri MMFF94 supportati si conservano
i conformeri ETKDG, con energia assente e fallback esplicito. Quando MMFF è
supportato si scartano le ottimizzazioni non convergenti o con energia non finita.
Numerosità effettive, eventuali conformeri scartati
e stato della minimizzazione vanno letti negli output dell'applicazione:
il numero richiesto non garantisce che tutte le conformazioni siano disponibili.
L'energia del ligando isolato non entra nello score di docking.
Grafo, atomi e loro ordine restano quelli dell'input. I vincoli tetraedrici ed E/Z
già definiti, anche dalla geometria 3D del MOL2, sono conservati su copie di lavoro
e verificati sulle coordinate prodotte. Le classi stereo più complesse sono
rifiutate esplicitamente.

La baseline cristallografica è separata ed esclusa dall'ensemble generato.
Proteina, target e indice geometrico sono condivisi. Il percorso predefinito
mantiene la ricerca CUDA sequenziale e il rescoring CPU originale.
Fast, WeightedInteraction e termini Vinardo alimentano quattro ranking
globali: Fast, WeightedInteraction, Vinardo rep50 e Hybrid.
Vinardo-rep50 usa alpha 0.5; Hybrid usa lambda 0.3 e alpha 0.5.
WI usa la regione pocket; Vinardo la proteina completa supportata.

Il pool predefinito contiene tutti i superstiti del clash. Il prefiltro esplicito
`--fast-prefilter 500` resta disponibile soltanto nel percorso CPU originale;
limita i tre ranking di rescoring, mentre Fast conserva tutti i superstiti.
Batch, overlap esplicito OFF/ON, rescoring CUDA e Oracle RMSD richiedono
`--fast-prefilter none`. Nessuna Top-1 Fast limita implicitamente il pool.
Hybrid è normalizzato e classificato sulla CPU una sola volta sull'intero pool
dell'ensemble, senza rinormalizzazione per conformero o batch. La baseline
ha una normalizzazione separata.

### Batch e overlap CPU–GPU

`--rescoring-backend cpu|cuda` sceglie il calcolo dei contributi grezzi (default
CPU). CUDA riusa gli scorer condivisi WeightedInteraction e Vinardo; Hybrid
globale resta distinto dal ranking per ligando del normale screening.

```sh
DOCKING_QUERY_THREADS=8 build/conformer-cuda/ligand_conformer_docking \
  /percorso/1a1e /percorso/nuovo-batch-overlap \
  --num-conformers 10 --seed 20261027 --min-rmsd 0.5 \
  --gpu-batch on --batch-size 4 --batch-memory-mib 2048 \
  --cpu-gpu-overlap on --query-method optimized \
  --rescoring-backend cuda --fast-prefilter none --pose-output topk --top-k 20
```

`--gpu-batch off` conserva ricerche singole; `on` aggrega ricerca e scoring CUDA.
`--batch-size` è il massimo desiderato (default 4, limite 64). Il budget esplicito
(default 2048 MiB) può suddividere un batch senza eliminare pose; un singolo
conformero troppo grande produce errore. Non è un limite alla VRAM totale.
`--cpu-gpu-overlap off|on` seleziona preparazione sincrona o un producer CPU con
coda limitata e un solo consumer CUDA. La baseline viene elaborata separatamente.
In batch la coda ha capacità `2 * batch-size` e un budget di 256 MiB per il
payload accodato stimato; senza batch conserva un solo packet pronto.
Una preparazione in corso resta fuori dal budget byte;
questo non limita la memoria totale del processo.
`--query-method reference|optimized` seleziona query equivalenti senza cambiarne
l'ordine; la CLI usa `optimized` se non sono specificati un flag o la variabile
`DOCKING_QUERY_METHOD`. Il flag ha precedenza sulla variabile; entrambi accettano
solo `reference` oppure `optimized`. Il percorso condiviso rispetta
`DOCKING_QUERY_THREADS` (default 1), con fallback seriale per input piccoli.
Il percorso originale mantiene un thread; `--gpu-batch on` richiede rescoring CUDA.

Gli identificatori conservano conformero e posa; le parità sono deterministiche.
RMSD usa corrispondenza atomica validata, heavy atoms e coordinate dirette,
senza riallineamento o correzione delle simmetrie: atomi equivalenti non vengono
scambiati. Riferimento assente o mapping incompatibile producono errore.
RMSD non partecipa alla selezione dei quattro ranking di scoring.
Gli output valutano Top-1/5/10/20 e l'oracle sui superstiti del clash, oltre
all'oracle dopo il prefiltro Fast. Non è un oracle sulle trasformazioni scartate.
`--pose-center-distance on` aggiunge `center_distance_A` alle sole pose esportate:
è la distanza in Å fra il centro geometrico degli atomi pesanti della posa e
quello del riferimento cristallografico, senza allineamento. È una diagnostica
di collocazione, distinta dall'RMSD; non interviene in ricerca, filtri o ranking.
Il default è `off`, senza cambiamenti ai CSV. I valori sono calcolati su richiesta
e riutilizzati tra metodi; `--pose-output none` non li calcola.
`--ranking rmsd` aggiunge ai quattro metodi un ordinamento diagnostico `rmsd`,
dal minimo RMSD, su tutti i superstiti dell'ensemble. La baseline è esclusa.
`local_rank` e `global_rank` identificano i ranghi per conformero e nell'ensemble;
la colonna `top1_minus_oracle_all_A` misura il divario fra la Top-1 selezionata
e il miglior RMSD disponibile. Per l'Oracle il divario è zero. Questa diagnosi
richiede la posa cristallografica e non è uno scoring predittivo.
`--crop-margin off` è il default; `8` attiva il crop chimico esplicito.
Gli SDF sono disattivati per default; `--write-sdf` abilita l'export molecolare.
`--pose-output none` disabilita anche gli SDF, quando richiesti.
Negli SDF raggruppati per conformero, `rank` è il rango locale del metodo;
il rango globale e l'RMSD sono nel CSV, collegati da conformero e `pose_id`.
La riproducibilità richiede anche la stessa versione RDKit, registrata nel protocollo.

| Output dell'app | Contenuto |
| --- | --- |
| `conformers.csv` | Metadati e ID baseline/ensemble del conformero, energia/diversità, conteggi della ricerca e del rescoring, oracoli e tempi per conformero. |
| `poses.csv` | Se abilitato, formato lungo con una riga per posa **per metodo**: ID, ranghi, score, RMSD e, nel formato completo, trasformazione. I termini di rescoring restano vuoti per pose Fast escluse dal prefiltro. |
| `summary.csv` | Top-1/5/10/20, successo, oracoli, divario Top-1/Oracle e tempi per popolazione: baseline e ensemble separati. |
| `protocol.csv` | Versione RDKit, seed, parametri, backend, regioni e tempi condivisi `baseline_execution_*` / `ensemble_execution_*`. |
| `conformers.sdf`, `structures/` | Solo con `--write-sdf` e output diverso da `none`: conformeri iniziali e pose raggruppate per popolazione/metodo/conformero; il default conserva Global Top-20. |

### Quanto esportare

Il default è `--pose-output all` con il formato completo.
Le opzioni agiscono dopo scoring, ranking e calcolo delle metriche: non
sono un prefiltro scientifico e non cambiano i pool o la normalizzazione Hybrid.

```sh
# Solo riepiloghi, metadati e metriche complete; nessun CSV per posa o SDF.
build/conformer-cuda/ligand_conformer_docking /percorso/1a1e /percorso/nuovo-riepilogo \
  --pose-output none

# Prime 20 pose di ogni popolazione/metodo; CSV senza matrici e traslazioni.
build/conformer-cuda/ligand_conformer_docking /percorso/1a1e /percorso/nuovo-top20 \
  --pose-output topk --top-k 20 --pose-detail compact
```

`none` non crea `poses.csv`, `conformers.sdf` o `structures/`. `topk` esporta
il prefisso globale di ciascun ranking per baseline e ensemble separatamente;
`all` esporta tutte le righe. `--top-k` è positivo e vale soltanto con `topk`.
`compact` conserva ID, ranghi, score, RMSD e clash; `full` aggiunge R/t come nel
formato predefinito. Riepiloghi e metriche usano sempre i pool completi.
Il protocollo conserva modalità, dettaglio, K, dimensioni dei file e tempo
di export, separato da ricerca e scoring.

Senza un `--pose-output` esplicito, `--write-sdf` conserva il comportamento
precedente: tutti i conformeri iniziali e solo le pose globali Top-20.
Con `--pose-output topk|all` esplicito, gli SDF delle pose seguono la stessa
quantità del CSV; `all --write-sdf` può produrre molti file e strutture.
Il wrapper `run_1a1e.py` propaga tutte queste opzioni.

Per un'analisi offline che debba riesaminare qualsiasi superstite servono
`--pose-output all --pose-detail full` e le coordinate esatte dei conformeri
(`--write-sdf`), oltre a riferimento e protocollo. `none`, `topk` e `compact`
non conservano quel materiale completo; il solo seed non ricostruisce in modo
affidabile il frame dei conformeri con versioni RDKit differenti. I vecchi
strumenti in `experiments/conformer_ensemble/` richiedono comunque il proprio
formato completo e non sono convertiti automaticamente da queste opzioni.

Con un solo complesso il successo Top-K è un indicatore 0/1, non una stima
del tasso di successo su un dataset.

## Diagnostica aggregata senza pose

`--diagnostics` genera una sola sequenza di 20 candidati ETKDGv3 con seed
20261027, diversità 0,5 Å e MMFF disattivato. I budget 1/5/10/20 sono prefissi
dei conformeri **accettati**, non nuove generazioni né numeri di tentativi.
Non si riempiono i budget incompleti. Ricerca e scoring vengono eseguiti una
volta per conformero e per l'originale; eventuali retry di memoria restano quelli
del percorso CUDA. I tempi misurano l'esecuzione massima, non run separati dei prefissi.

```sh
DOCKING_QUERY_THREADS=1 build/conformer-cuda/ligand_conformer_docking \
  /percorso/1a1e /percorso/nuova-diagnostica \
  --diagnostics --num-conformers 20 --seed 20261027 --min-rmsd 0.5 \
  --crop-margin 8 --cutoff 24 --query-method optimized \
  --gpu-batch on --batch-size 4 --batch-memory-mib 2048 \
  --rescoring-backend cuda --fast-prefilter none --pose-output none
```

Omettere `--cpu-gpu-overlap` per il percorso senza overlap intenzionale;
non usare `off` esplicito per questo protocollo. Il budget di memoria limita
i buffer GPU espliciti, non tutta la VRAM o la RAM dei survivors conservati.

Per ogni budget si analizzano originale incluso/escluso e, per ciascuna
condizione, pool `full` e `fast_top500` globale. I cinque ranking sono Fast,
WeightedInteraction, Vinardo standard (alpha 1), rep50 (alpha 0,5) e Hybrid
(alpha 0,5, lambda 0,3). Gli score grezzi vengono riutilizzati; Hybrid viene
rinormalizzato su ogni pool effettivo. L'RMSD non viene usato per scegliere pose.
Le parità seguono `(conformer_id, pose_id)`; l'originale ha ID 0 e può quindi
precedere altri conformeri a score identico. Il summary conta questi pareggi.

L'originale è la geometria del riferimento cristallografico, sottoposta al
normale docking: la posa nativa non viene inserita direttamente tra i candidati.
L'inclusione è una condizione di redocking con geometria interna privilegiata.
Anche i conformeri generati possono ereditare vincoli stereochimici ricavati
dall'input 3D prima della rimozione delle coordinate; non è un protocollo privo
di ogni informazione geometrica nativa. La policy di generazione resta invariata.

Questa modalità non chiama i writer delle pose e rifiuta `--write-sdf` o export
`topk/all`. Le metriche vengono calcolate in memoria anche senza export.
Non è compatibile con `--original-only`, `--generate-only`, `--ranking rmsd`
o con un prefiltro per conformero. Senza `--diagnostics` nulla cambia.

| File | Contenuto |
| --- | --- |
| `summary.csv` | 80 righe: budget, inclusione originale, pool, metodo, conteggi, Oracle RMSD/centroide e Top-1 con ID e successi. |
| `topk_metrics.csv` | 1.600 righe: K=1..20, disponibilità e minimi indipendenti RMSD/centroide con ID e ranghi. |
| `conformers.csv` | Statistiche per conformero, ID ETKDG e `accepted_order`; nessuna coordinata. L'originale non ha ordine di accettazione. |
| `protocol.csv` | Configurazione, RDKit, informazione nativa, pool, soglie, conteggi di generazione e tempi. |
| `status.csv` | Completamento, fase, errore ed eventuali contatori disponibili. |

`global_pose_id` è un ordinale zero-based nell'unione di tutti i survivors,
ordinati per conformero/posa, e resta stabile fra condizioni e budget.
`budget_available` include anche conformeri accettati senza pose;
`budget_complete` distingue i prefissi incompleti. I minimi Top-K conservano
separatamente le pose e i ranghi che minimizzano RMSD e distanza del centroide.
I successi usano confronti stretti `<2`, `<3` e `<5 Å`. Pool vuoti hanno stato
`no_poses` e metriche vuote; errori di preparazione o GPU interrompono il
complesso e non causano esclusioni silenziose. I conteggi non ricostruibili
dopo un errore restano vuoti. Un run completato ha `completed=1` in `status.csv`.

Il cutoff CUDA resta fisso, senza il fallback al diametro del driver joint CPU.
I due protocolli non hanno necessariamente lo stesso pool. Senza strutture e
trasformazioni salvate, nuove analisi non previste richiedono una riesecuzione.

## Quattro prove singole: 1/5/10/20

Eseguire il wrapper nell'ambiente CUDA:

```sh
python3 -B experiments/ligand_conformer_docking/run_1a1e.py \
  --complex /percorso/1a1e \
  --executable /percorso/build/conformer-cuda/ligand_conformer_docking \
  --output /percorso/nuova-serie
```

Il wrapper avvia quattro processi distinti, una volta ciascuno, richiedendo
1, 5, 10 e 20 conformeri con gli stessi parametri. Non riutilizza il prefisso
di una sola generazione da 20. Seed, MMFF, diversità, prefiltro, crop, cutoff,
input, output, batch, overlap, query e backend sono configurabili e restano
costanti nella serie. Il wrapper passa sempre `--query-method reference`, salvo
un override esplicito a `optimized`, mantenendo il proprio protocollo fisso.
Batch, overlap e backend vengono inoltrati solo quando specificati; senza di
essi resta il percorso sequenziale con rescoring CPU.
La CLI completa è disponibile con `run_1a1e.py --help`.

Ogni `conformers-N/` contiene comando, stdout, stderr e `docking/` dell'app.
`protocol.json` registra comandi, ambiente selezionato, Git HEAD/branch/stato/diff
e stato della serie, più SHA-256 dell'eseguibile e dei `.cpp`, `.hpp`, `.py` e
`CMakeLists.txt` della suite.
`runs.csv` riporta stato, exit code e wall esterno per prova.
Il wall comprende avvio, generazione, ricerca, scoring, RMSD e I/O/export;
sono quattro osservazioni singole, non medie di un benchmark ripetuto.
Non si sommano timer annidati né tempi CPU/GPU sovrapposti. Le ricerche e gli
scoring condivisi sono vuoti nei CSV per conformero e nei totali di popolazione;
i tempi effettivi sono nel protocollo, separati fra baseline ed ensemble.
`*_execution_wall_s` include preparazione dei ligandi, attese, ricerca, scoring
e RMSD; esclude recettore, ETKDG, ranking globale ed export.
I totali di fase dell'adapter includono anche il lavoro dei tentativi adattivi
rifiutati. Sono intervalli host completati, non tempi isolati dei kernel.
Il download completo dei superstiti appartiene alla ricerca con rescoring CPU,
al rescoring grezzo con CUDA. Confrontare il wall complessivo fra backend;
confrontare le singole fasi solo con lo stesso backend e protocollo.
`--original-only` permette una misura end-to-end separata della sola geometria
originale, senza generazione ETKDG.
Gli input scientifici non vengono copiati né sottoposti a checksum dal wrapper.

La directory radice deve essere nuova. Un exit code nonzero o l'assenza di
`summary.csv` interrompe la serie e conserva log e output parziali.
`protocol.json` diventa `complete` solo dopo quattro esecuzioni riuscite con
riepilogo presente; le configurazioni non eseguite non producono risultati.

Test del solo wrapper, con tutti i processi simulati e nessun dataset aperto:

```sh
python3 -B -m unittest discover -s experiments/ligand_conformer_docking \
  -p 'test_run_1a1e.py' -v
```
