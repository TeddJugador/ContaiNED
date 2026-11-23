ContaiNED: Type-Constrained Mention Clustering for Zero-Shot Named Entity Disambiguation

Honours research project, Department of Computer Science, Rhodes University (2026).
Author: Theodore Masi · Supervisor: Marc Marais

ContaiNED is a zero-shot named entity disambiguation (NED) pipeline. It links ambiguous mentions in text (Amazon the company or Amazon the forest?) to Wikipedia entities without training or fine-tuning any model and without annotated data from the target domain.

Instead of resolving each mention on its own, ContaiNED groups mentions that refer to the same entity within a document, pools their surrounding context, and makes one decision per group. A container class mechanism uses predicted entity types (person, organisation, location, misc) to stop mentions of different types from being grouped together.

▶ Run it on Kaggle (free GPU): https://www.kaggle.com/code/theomasi/contained

How it works

flowchart LR
    A[Document] --> B[NER tagging]
    B --> C[Mention clustering<br/>with type guard]
    C --> D[Candidate generation]
    D --> E[Context pooling]
    E --> F[BERT scoring]
    F --> G[Hybrid ranking]
    G --> H[Predicted entity]

	1.	NER tagging. dslim/bert-base-NER predicts a coarse type (PER, ORG, LOC, MISC) for each gold mention.
	2.	Mention clustering. Mentions are grouped longest-first, merging when one name contains the other or their Levenshtein ratio is at least 0.85. In the container class configuration, mentions of different predicted types are never merged.
	3.	Candidate generation. Candidates come from the ZELDA alias table (exact, case-insensitive, then core-name match), with live Wikipedia search as a fallback.
	4.	Context pooling. Each mention contributes a 200-character window on either side, and windows in a group are concatenated.
	5.	Hybrid ranking. Candidates are scored as 0.40 × semantic + 0.35 × string + 0.25 × prior, where semantic similarity comes from the frozen sentence-transformers/bert-base-nli-mean-tokens encoder.

The only difference between the baseline and container class configurations is the type guard in step 2.

Repository contents

File	Contents
contained.ipynb	The full pipeline and all three experiments
requirements.txt	Python dependencies
CITATION.cff	Citation metadata
LICENSE	MIT licence

Running it

The easiest way is the Kaggle notebook, which already has the data attached and a free GPU.

To run it elsewhere:

	1.	Install the dependencies: pip install -r requirements.txt
	2.	Download the ZELDA test splits and candidate lists from the ZELDA repository.
	3.	Update ZELDA_DIR, CANDIDATE_PICKLE and IDS_PICKLE near the top of the notebook to point at your copies.
	4.	Run the smoke test cell before the full evaluation.

The full evaluation of both configurations took about 7.7 hours on a single NVIDIA Tesla T4 (Python 3.12, PyTorch 2.10, Transformers 5.0). Entity descriptions and tagger output are cached to disk, so both configurations see identical candidates.

Limitations

	•	The NER tagger was trained on edited newswire and was not adapted, so the type constraint is only as reliable as the tagger is on a given kind of text.
	•	Hyperparameters were chosen on 45 documents drawn from the test splits (0.93% of the evaluation data).
	•	Pooled contexts are truncated to 128 tokens, which limits the benefit of pooling for large groups.

Citation

If you use this work, please cite it using the metadata in CITATION.cff.

Acknowledgements

This work was undertaken in the Distributed Multimedia CoE at Rhodes University. Evaluation uses the ZELDA benchmark (Milich and Akbik, 2023).