# Docking CUDA di un ensemble di conformeri — WIP

Esperimento C++ indipendente sul branch `exp/ligand-conformer-ensemble`.
Riusa le API del motore senza modificare `gpu_docking`, i ranking scientifici
o il precedente WIP in `experiments/conformer_ensemble/`. La validazione CUDA
su GPU e le prove reali 1/5/10/20 restano da eseguire: questi comandi non
costituiscono risultati sperimentali già ottenuti.

## Compilazione e controlli host

Il generatore usa RDKit C++, senza Python RDKit o subprocessi per ETKDG/MMFF.
Servono le dipendenze C++ già previste dal progetto e i componenti RDKit di
embedding, force field e allineamento. Nessuna installazione è automatica.

```sh
cmake -S experiments/ligand_conformer_docking -B build/conformer-host \
  -DENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DTARGET_CPU_ARCHITECTURE=generic
cmake --build build/conformer-host \
  --target ligand_conformer_docking conformer_generation_test ensemble_ranking_test --parallel 1
ctest --test-dir build/conformer-host \
  -R 'conformer_generation|ensemble_ranking' --output-on-failure
```

La build host supporta `--generate-only` e i test sintetici, non la ricerca CUDA.
Per A100, mantenendo le impostazioni delle dipendenze del proprio ambiente:

```sh
cmake -S experiments/ligand_conformer_docking -B build/conformer-cuda \
  -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DTARGET_CPU_ARCHITECTURE=generic
cmake --build build/conformer-cuda --target ligand_conformer_docking --parallel 1
```

Il test opzionale `conformer_original` richiede CUDA e un complesso autorizzato,
indicato in configurazione con `-DDOCKING_TEST_COMPLEX=/percorso/complesso`.
Non viene eseguito automaticamente dal wrapper Python.

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
rifiutate esplicitamente da questa prima versione.

La baseline cristallografica è separata ed esclusa dall'ensemble generato.
Ogni conformero viene preparato e cercato riusando un unico `PreparedGpuTarget`
e `run_gpu_ligand_search`: ricerca CUDA, rescoring sul CPU con le API condivise.
Fast, WeightedInteraction e termini Vinardo vengono riusati per quattro ranking
globali: Fast, WeightedInteraction, vinardo-rep50 e Hybrid.
Vinardo-rep50 usa alpha 0.5; Hybrid usa lambda 0.3 e alpha 0.5.
WI usa la regione pocket; Vinardo la proteina completa supportata.

Di default si classifica il pool completo dei superstiti del clash.
`--fast-prefilter 500` limita per conformero il pool degli altri tre ranking;
Fast globale continua a usare tutti i superstiti. Hybrid normalizza una sola
volta sull'unione del pool dei conformeri generati, prima di qualunque Top-K.
La baseline ha un pool e una normalizzazione separati. Non si confrontano
numericamente Hybrid normalizzati su popolazioni diverse.

Gli identificatori conservano conformero e posa; le parità sono deterministiche.
RMSD usa corrispondenza atomica validata, heavy atoms e coordinate dirette,
senza riallineamento o correzione delle simmetrie; non partecipa alla selezione.
Gli output valutano Top-1/5/10/20 e l'oracle sui superstiti del clash, oltre
all'oracle dopo il prefiltro Fast. Non è un oracle sulle trasformazioni scartate.
`--crop-margin off` è il default; `8` attiva il crop chimico esplicito.
Gli SDF sono disattivati per default; `--write-sdf` abilita l'export molecolare.
Negli SDF raggruppati per conformero, `rank` è il rango locale del metodo;
il rango globale e l'RMSD sono nel CSV, collegati da conformero e `pose_id`.
La riproducibilità richiede anche la stessa versione RDKit, registrata nel protocollo.

| Output dell'app | Contenuto |
| --- | --- |
| `conformers.csv` | Metadati e ID baseline/ensemble del conformero, energia/diversità, conteggi della ricerca e del rescoring, oracoli e tempi per conformero. |
| `poses.csv` | Formato lungo: una riga per posa **per metodo**, con ID conformero/posa/allineamento, ranghi locale/globale, quattro score, RMSD e trasformazione. I termini di rescoring restano vuoti per pose Fast escluse dal prefiltro. |
| `summary.csv` | Top-1/5/10/20, successo, oracoli e tempi aggregati per popolazione: baseline e ensemble separati. |
| `protocol.csv` | Coppie chiave/valore: versione RDKit, seed, parametri, regioni, policy e ambito dei clock. |
| `conformers.sdf`, `structures/` | Solo con `--write-sdf`: conformeri iniziali e strutture Global Top-20, raggruppate per popolazione/metodo/conformero. |

Con un solo complesso il successo Top-K è un indicatore 0/1, non una stima
del tasso di successo su un dataset.

## Quattro prove singole: 1/5/10/20

Eseguire esplicitamente il wrapper sul nodo GPU autorizzato:

```sh
python3 -B experiments/ligand_conformer_docking/run_1a1e.py \
  --complex /percorso/1a1e \
  --executable /percorso/build/conformer-cuda/ligand_conformer_docking \
  --output /percorso/nuova-serie
```

Il wrapper avvia quattro processi distinti, una volta ciascuno, richiedendo
1, 5, 10 e 20 conformeri con gli stessi parametri. Non riutilizza il prefisso
di una sola generazione da 20. Seed, MMFF, diversità, prefiltro, crop, cutoff,
input alternativi e SDF sono configurabili e restano costanti nella serie.
La CLI completa è disponibile con `run_1a1e.py --help`.

Ogni `conformers-N/` contiene comando, stdout, stderr e `docking/` dell'app.
`protocol.json` registra comandi, ambiente selezionato, Git HEAD/branch/stato/diff
e stato della serie, più SHA-256 dell'eseguibile e dei `.cpp`, `.hpp`, `.py` e
`CMakeLists.txt` della nuova directory sperimentale, anche prima del commit.
`runs.csv` riporta stato, exit code e wall esterno per prova.
Il wall comprende avvio, generazione, ricerca, scoring, RMSD e I/O/export;
sono quattro osservazioni singole, non medie di un benchmark ripetuto.
Non si sommano timer annidati e non si omette ETKDG dal costo end-to-end.
Il wall dell'applicazione comprende insieme baseline ed ensemble; i tempi per
popolazione sono distinti nel summary. `--original-only` permette una misura
end-to-end separata della sola geometria originale, senza generazione ETKDG.
Gli input scientifici non vengono copiati né sottoposti a checksum dal wrapper.

La directory radice deve essere nuova. Un exit code nonzero o l'assenza di
`summary.csv` interrompe la serie, conserva log/output parziali e non inventa
risultati per le configurazioni non eseguite. `protocol.json` diventa `complete`
solo dopo quattro exit code zero con summary presente; il wrapper non dipende
dai campi interni dei CSV dell'app, ancora in evoluzione.

Test del solo wrapper, con tutti i processi simulati e nessun dataset aperto:

```sh
python3 -B -m unittest discover -s experiments/ligand_conformer_docking \
  -p 'test_run_1a1e.py' -v
```
