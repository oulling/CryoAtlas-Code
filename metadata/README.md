# Dataset metadata

[Croissant_File.json](Croissant_File.json) describes the complete public CryoAtlas v1.0 release on Hugging Face using Croissant 1.1 and RAI 1.0. It includes the 70 training/test archives, the complete benchmark resource package, typed catalogs and metrics, native scientific file collections, nine creators, and CC BY-NC 4.0 licensing.

All data-file URLs are pinned to the HF snapshot recorded in the metadata. Benchmark targets and examples are explicitly identified as subsets of the fixed train/test release.

[croissant_validation.json](croissant_validation.json) summarizes the passed MLCommons schema validation, the NeurIPS-recommended online checker's JSON/schema/RAI/record-generation tests, 247 publicly accessible file URLs, and native-file coverage checks. Large native map/CIF/FASTA resources are described as FileSets; automatic record-loading tests use the seven lightweight entry, archive and result tables.

The byte-identical metadata and validation summary are also available [with the public data](https://huggingface.co/datasets/markfaa/CryoAtlas/tree/main/resources/metadata). To use this release in OpenReview, attach Croissant_File.json and supply the [HF dataset URL](https://huggingface.co/datasets/markfaa/CryoAtlas).
