# Ablation di WeightedInteraction e Vinardo-like

Questo esperimento ricalcola lo scoring su pose già generate. Non modifica il
motore e non ripete lookup, Kabsch, clash o ricerca geometrica durante l'ablation.
Il pool rimane fisso: nessuna posa viene scartata in base a WeightedInteraction, Vinardo o RMSD.
Non usa soltanto le Top-10 e non applica un nuovo prefilter Fast.

## Cosa viene riutilizzato

`score_weighted_interactions()` accetta già pesi diversi per le sette famiglie:
hydrophobic, pi stacking, hydrogen bond, halogen bond, cation-pi, ionic e metal.
`VinardoLikeScorer::score()` restituisce già le quattro componenti firmate e
ponderate: gaussian, repulsion, hydrophobic e hydrogen bond. Non sono termini
grezzi ai quali moltiplicare nuovamente i coefficienti.

Il tool usa questi scorer, `rank_rescored_poses()` e il helper host
`gpu_pose_heavy_atom_rmsd()`. Non contiene copie delle formule di interazione.
La build è CPU e non richiede una GPU. Il target `docking_gpu_output` usato per
RMSD è anch'esso codice host; il nome non implica un'esecuzione CUDA.

## Compilazione isolata nel container

Nel container con le dipendenze C++ già usate dal progetto:

```sh
cd /workspace/geometric-docking
cmake -S experiments/scoring_ablation -B experiments/scoring_ablation/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build experiments/scoring_ablation/build \
  --target scoring_ablation scoring_ablation_test vinardo_rescore_experiment --parallel 1
ctest --test-dir experiments/scoring_ablation/build --output-on-failure \
  -R '^scoring_ablation_'
```

Il CMake locale richiama i target esistenti in una nuova directory di build,
con CUDA disattivato. Non modifica il CMake principale né le build precedenti.
Il generatore esistente viene compilato in `build/engine/` soltanto per avere
un eseguibile corrispondente agli stessi sorgenti degli scorer.
La dipendenza attuale imposta esplicitamente `CMP0167 NEW`: serve quindi un
CMake che riconosca quella policy (3.30 o successivo), oltre alle dipendenze già
richieste dal progetto. Un default `CMAKE_POLICY_DEFAULT_CMP0167` non sostituisce
quell'impostazione esplicita. Eventuali problemi dell'ambiente vanno segnalati,
senza modificare la dipendenza per questo esperimento.

## Prima prova su 1a1e

Non essendo disponibile un CSV completo preesistente, il runner genera una sola
volta tutte le pose tramite il percorso `--joint-output` già implementato in
`vinardo_rescore_experiment`, senza `--prefilter`. Poi verifica che il numero di
righe coincida con `input_survivors` e `exported_poses` del protocollo, e avvia
l'ablation senza ulteriori docking. Il ligando cristallografico viene usato
come input rigido e, separatamente, come riferimento della metrica RMSD.

```sh
cd /workspace/geometric-docking
bash experiments/scoring_ablation/run_1a1e.sh /workspace/datasets/refined-set/1a1e
```

La directory nuova è `experiments/scoring_ablation/runs/1a1e-<data>-<pid>/`.
Non si sovrascrivono esecuzioni precedenti. Si possono indicare come secondo
argomento una directory nuova e come terzo una diversa build dell'esperimento.
Il runner non compila automaticamente. I suoi input sono soltanto i tre file
`1a1e_protein.pdb`, `1a1e_pocket.pdb` e `1a1e_ligand.mol2`, letti senza modificarli.

La generazione usa cutoff 24 Å, quaternion e clash cached, con profiling
disattivato. Quel driver conserva la propria politica storica: se il diametro
del ligando supera 24 Å usa un indice specifico. Il protocollo registra la
modalità effettiva. Non va chiamato automaticamente screening shared-only;
questa distinzione non viene cambiata per eseguire l'ablation.

`generation/poses.csv` conserva l'intero pool, gli ID e tutte le trasformazioni
a 17 cifre: ogni posa è rappresentata dal ligando originale e dalla sua matrice
R e traslazione t. `generation/protocol.csv` conserva impostazioni, conteggi e
tempi della generazione. Gli originali non vengono copiati o trasformati sul
disco. `commands.txt`, stdout e stderr dei due programmi restano nella nuova
directory. Non si producono percentuali scientifiche finché la prova non è stata
eseguita e verificata.

## Rianalizzare lo stesso pool

Per cambiare soltanto alpha o lambda non va ripetuta la generazione. Esempio,
dopo avere impostato `RUN` alla directory appena prodotta:

```sh
POOL_SIZE=$(python3 -c 'import csv,sys; p=dict(csv.reader(open(sys.argv[1], newline=""))); print(p["exported_poses"])' "$RUN/generation/protocol.csv")
experiments/scoring_ablation/build/scoring_ablation \
  --complex /workspace/datasets/refined-set/1a1e \
  --poses "$RUN/generation/poses.csv" --pool-size "$POOL_SIZE" \
  --pool-kind all-survivors --interaction-region pocket \
  --reference /workspace/datasets/refined-set/1a1e/1a1e_ligand.mol2 \
  --vinardo-alpha 0.5 --hybrid-lambda 0.3 \
  --output "$RUN/ablation-rep50"
```

Il tool legge anche il CSV delle pose di `gpu_docking`, con la colonna
`fast_weighted_score`. Usa `pose_id` se presente, altrimenti `alignment_index`
come ID stabile. `--record N` seleziona un singolo record; con ligandi diversi
occorre anche `--ligand FILE` corrispondente a quel record. Non aggregare pool di
ligandi o conformeri diversi come se fossero un singolo complesso.

`--pool-size` è obbligatorio e deve provenire dal protocollo di generazione.
Il controllo delle righe rileva troncamenti rispetto a quel numero; da solo non
può dimostrare che un export fosse completo. Dichiarare `--pool-kind
fast-prefilter` o `exported` se il pool era già limitato. L'ablation rimane valida
su quel pool, ma le conclusioni sono condizionate alla selezione precedente.
Il runner incluso controlla esplicitamente l'assenza di prefilter e la completezza
rispetto ai superstiti del generatore.

WeightedInteraction e Vinardo vengono ricalcolati dalla molecola originale e dalle
trasformazioni. Se i rispettivi punteggi sono presenti nel CSV, devono coincidere
con quelli salvati entro `1e-6 * (1 + abs(score_salvato))`, altrimenti il tool
si ferma prima di scrivere i risultati. `--score-tolerance` rende esplicita una
tolleranza diversa; non va allargata per nascondere un disaccordo di configurazione.
Il protocollo conta quanti punteggi salvati sono stati effettivamente verificati.
Se mancavano, la baseline è un nuovo rescoring CPU e non un'equivalenza dimostrata
con uno scoring precedente. Fast viene conservato dal CSV e non ricalcolato.

## Perché WeightedInteraction richiede un vero rescoring

WeightedInteraction somma, per ogni area fisica del ligando, il massimo fra zero e i pesi dei
contatti riconosciuti su quell'area. Due contatti della stessa area non producono
necessariamente due premi. Per esempio, un'area con ionic=3 e cation-pi=2,5
contribuisce 3; eliminando ionic può ancora contribuire 2,5.

Per ogni famiglia il tool azzera soltanto quel peso e richiama lo scorer
esistente, permettendo il subentro degli altri contatti. Non sottrae
`conteggio * peso`. Contatti e coperture restano quelli geometrici: si elimina
il premio, non l'esistenza dell'interazione. La perdita marginale non è una
decomposizione additiva, può essere zero in presenza di premi alternativi e
non va sommata alle perdite delle altre famiglie per ricostruire il totale.

Per Vinardo vengono azzerate una alla volta le componenti ponderate, risommando
quelle rimaste nello stesso ordine dello scorer. Se alpha è 0,5, eliminare la
repulsione elimina l'intero contributo repulsivo ridotto; non lo sottrae due volte.
Vinardo resta un prototipo Vinardo-like, non un'affinità validata in kcal/mol.

WeightedInteraction valuta per default le aree appartenenti alla pocket secondo il criterio
standard di `RescoringTarget`. `--interaction-region protein` usa tutte le aree. Nessun
crop a 8 Å viene introdotto. Vinardo usa sempre la proteina completa supportata
e il cutoff atomico stretto di 8 Å già presente nello scorer. Un CSV generato
con crop o pesi diversi non è automaticamente compatibile con questa baseline.

## Ranking, RMSD e interpretazione

WeightedInteraction e Hybrid vengono massimizzati; Vinardo viene minimizzato. Hybrid usa il
ranker esistente, con `H = lambda*z(WeightedInteraction) + (1-lambda)*z(-V_alpha)`.
Media e deviazione standard della popolazione vengono ricalcolate **per ogni
ablation sul medesimo pool completo**; una componente costante ha z-score zero.
Non si riusano le statistiche della baseline. Alpha e lambda sono fissati prima
dell'esperimento, non scelti guardando l'RMSD di 1a1e.

Le parità seguono `pose_id` crescente e poi ordine d'ingresso. È il criterio
di `rank_rescored_poses()`, non quello del vecchio `rank_poses()` con copertura,
numero di interazioni e clash come ulteriori tie-break. La Top-1 originale
nel confronto è la baseline ricalcolata con questo ranker sul pool dichiarato.

Il riferimento deve avere gli stessi indici atomici, elementi, cariche e grafo,
compresi gli H espliciti. Il helper del motore verifica questa corrispondenza.
L'RMSD riguarda soltanto gli atomi pesanti, nel sistema della proteina, senza
riallineamento o correzione delle simmetrie. Il successo è `RMSD < 2 Å`.
Questi dati vengono aggiunti dopo il ranking: non scelgono o ordinano le pose.
Senza riferimento restano vuoti, con stato `no_reference`; un pool vuoto ha
stato `no_poses`, mentre errori di lettura o mapping causano un errore esplicito.

| File in `ablation/` | Contenuto |
| --- | --- |
| `contributions.csv` | Una riga per posa: Fast, WeightedInteraction, contatti/coperture, quattro componenti Vinardo, totale standard e corretto con alpha, RMSD opzionale. |
| `weighted_interaction_ablation.csv` | Sette righe per posa: famiglia, conteggio, peso predefinito, score senza quella famiglia e perdita marginale. |
| `rankings.csv` | Tutte le pose e tutti i ranghi per baseline e ablation; nessun taglio Top-k. |
| `top1_changes.csv` | Top-1 prima/dopo, cambio di ID, nuovo rango della precedente Top-1, RMSD e successo. |
| `protocol.csv` | Input, pool, regioni, controlli sui punteggi, alpha/lambda, tie-break, convenzioni e tempi del replay. |

Un termine numericamente grande non è necessariamente quello che cambia la
Top-1: può essere quasi costante sul pool. Molti contatti WeightedInteraction possono essere
ridondanti. Un cambio di ID può inoltre avvenire fra pose geometricamente simili;
va letto insieme a RMSD e punteggi, non come miglioramento automatico.
Una sola ablation su 1a1e è un controllo esplorativo, non una stima generale
dell'importanza dei termini. Per pochi altri complessi si ripete il runner,
senza cambiare simultaneamente campionamento, scoring e parametri.

## Controlli predisposti e stato della verifica

Il test sintetico copre il massimo WeightedInteraction fra due famiglie sulla stessa area,
componenti Vinardo e alpha, cancellazione numerica, pool fisso e nuova Top-1
oltre le prime dieci, rinormalizzazione Hybrid, parità, pool costanti/vuoti e
lettura CSV con ID duplicati, campi quotati, record multipli e valori non finiti.
Un test Python separato verifica il runner con input sintetici ed eseguibili
fittizi: output esistenti, prefilter, pool troncati, ID duplicati e riferimento.
Non esegue scoring né apre dati scientifici. Può essere eseguito anche senza RDKit:

```sh
python3 experiments/scoring_ablation/workflow_test.py
```

Durante questa implementazione non sono stati compilati o eseguiti i test C++
né il docking reale. Gli eseguibili preesistenti controllati sul Mac sono ELF
Linux aarch64 e non sono eseguibili macOS; non risultano header RDKit nei percorsi
Homebrew standard controllati. La prova minima è quindi predisposta per il
container, senza tentare di accedervi o installare dipendenze.
Sono invece passati i cinque test Python del flusso con dati sintetici, il
controllo sintattico Bash del runner e `git diff --check`. Questi controlli non
validano la compilazione C++ né dimostrano un risultato scientifico su 1a1e.
