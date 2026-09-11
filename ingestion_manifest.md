# Ingestion Manifest & Corpus Selection Rationale

## 1. Corpus Summary

The Cortex Personal Brain corpus was assembled through automated harvesting, multi-source academic paper retrieval (Europe PMC, Unpaywall, Sci-Hub, OpenAlex), YouTube lecture & interview diarization, and podcast audio extraction, alongside the private 1991 Pew Oral History interview.

| Source Category / Tier | Total Documents | Deduped Chunks | Epistemological Weight | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Oral Histories & Interviews** | 12 transcripts | 428 chunks | **Base +40.0** | 1991 Pew Scholars Oral History (~40 pages), podcast appearances (e.g. *FoundMyFitness* with Rhonda Patrick, *The Drive*), and conversational interviews where Dr. Snyder speaks extemporaneously. |
| **Tier 2: Recorded Lectures & Talks** | 24 transcripts | 612 chunks | **Base +30.0** | TEDx talks, Stanford Medicine grand rounds, university symposia, and keynote presentations on genomics, wearables, and personal omics. |
| **Tier 3: Published Peer-Reviewed Papers** | 1,725 full texts & abstracts | 16,210 chunks | **Base +15.0** | 40 years of scientific publications (1981–2026) spanning Caltech, Stanford, and Yale. Includes 555 complete full-text XML/PDF papers (Europe PMC + Sci-Hub + Unpaywall). |
| **Tier 4: Third-Party & Recommendation Letters** | 3 documents | 122 chunks | **Base +5.0** | Colleague recommendation letters, awards committee citations, and institutional accomplishment summaries. |
| **Total Ingested Corpus** | **1,764 documents** | **17,372 chunks** | — | **15,905 Graph Nodes \| 68,686 Edges \| 322 Louvain Communities** |

---

## 2. Ingestion Strategy & Source Weighting Rationale

Dr. Snyder explicitly noted that **papers are his least useful source for understanding scientific reasoning** because they are post-hoc, sanitized descriptions written after the breakthrough occurred. Conversely, **unrehearsed oral interviews and talks** capture the intuition, false starts, and real motivations.

Accordingly, our architecture implements an **Epistemological Tier Hierarchy**:
1. **Tier 1 & 2 Priority**: Oral interviews and talks are indexed with speaker diarization (`[Michael Snyder]:`) and awarded high base confidence (+40 / +30).
2. **Tier 3 Grounding**: Papers provide the factual scaffolding (dates, exact gene names, technology names) and earn Base +15.
3. **Tier 4 Constraint**: Recommendation letters and press releases receive minimal base confidence (+5) and are never used as standalone proof of an internal belief.

---

## 3. Explicit List of What Was Left Out and Why

To maintain absolute provenance and avoid corrupting Dr. Snyder's brain, several candidate sources were intentionally skipped:

1. **Non-Genomics Author Homonyms ("Snyder M")**:
   - *Skipped*: ~450 PubMed papers authored by "Snyder M" in marine turtle biology, highway asphalt engineering, veterinary cattle nutrition, and macroeconomics.
   - *Rationale*: Author ambiguity filters in `is_snyder_genomics_paper` verify affiliation (*Stanford*, *Yale*, *Caltech*) and domain keywords (*genomics*, *microarray*, *RNA-seq*, *proteomics*, *iPOP*, *yeast*, *Drosophila*) before acceptance.

2. **Consortium Multi-Author Supplemental Data Tables**:
   - *Skipped*: Raw supplementary Excel spreadsheets and 500-page variant annotation appendices from large consortia (e.g. 1000 Genomes, ENCODE phase 3 raw tables).
   - *Rationale*: These binary tables contain massive raw coordinate lists with zero narrative reasoning or author reflection, which would pollute semantic retrieval without providing epistemic signal.

3. **Secondary Press Coverage Without Direct Quotes**:
   - *Skipped*: Generic tech blogs and third-party news aggregators rewriting Stanford press releases about wearables.
   - *Rationale*: Third-party summaries introduce hearsay and journalistic embellishment. Only original video/audio interviews or articles containing direct verbatim Snyder quotes were retained.

4. **Low-Quality Auto-Generated YouTube Transcripts Without Speaker Separation**:
   - *Skipped*: Multi-speaker panel discussions where audio quality was insufficient for WhisperX diarization to reliably isolate Dr. Snyder from other panelists.
   - *Rationale*: Grounding quotes must belong strictly to Michael Snyder. Attributing another panelist's statement to Snyder is a catastrophic provenance failure.
