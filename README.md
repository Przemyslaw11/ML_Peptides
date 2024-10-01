# PeptiGraph: Molecular Fingerprints for Peptide Classification

It is a repository with the source code for the project of the 2024 class of the Advanced Machine Learning course at the AGH University of Krakow. The scope of the project is to classify peptides using molecular fingerprints and compare their performance with ProtBERT and protein descriptors from PyBioMed.

## Classification of Peptides Using Molecular Methods

Peptides are small proteins that play key roles in living organisms. Moreover, like all proteins, they often serve multiple functions simultaneously, influencing their higher-level characteristics. Due to their relatively small size, they can be processed with a reasonable computational cost, which is often a challenge for larger proteins.

Until now, peptides have not typically been treated on a large scale simply as molecular graphs. Instead, dedicated algorithms, which typically operate on amino acid sequences, are used. The question is whether using molecular fingerprints to vectorize molecules, which operate directly on molecular graphs, will yield good results. This is a very low-level and detailed representation.

## Project elements:

- Collect datasets for peptide classification
- Test molecular fingerprints from the scikit-fingerprints library for peptide classification
- Conduct a comparison with, for example, ProtBERT or descriptors from PyBioMed

## Expected results:

- Benchmarking molecular fingerprints for peptide classification
- Comparison with ProtBERT and protein descriptors
