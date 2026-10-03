# Third-Party Notices

This repository includes third-party source snapshots and derived model code. Each component retains its original license and applicable copyright notices. This document summarizes the bundled components; it does not replace their license texts or file-level notices.

## Bundled Dependency Sources

| Component | Declared version | Recorded snapshot revision | License scope | Archive |
|---|---|---|---|---|
| [Transformers](https://github.com/huggingface/transformers) | `5.7.0` | `90dd8674248257e1115e531f2250fd85a32c3863` | Apache-2.0 | [`transformers.tar.gz`](third_party/transformers.tar.gz) |
| [SGLang](https://github.com/sgl-project/sglang) | `0.5.6.post2` | `a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b` | Apache-2.0 | [`sglang.tar.gz`](third_party/sglang.tar.gz) |
| [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) | `0.5.0` | `bbc1f57e892a6e5af586c71c7733411208c75b82` | MIT (evaluation pipeline); Apache-2.0 (multimodal models and tasks) | [`lmms_eval.tar.gz`](third_party/lmms_eval.tar.gz) |

The versions above are reported by the bundled sources. These are adapted snapshots and may differ from packages with the same version on a package index. Snapshot revision identifiers describe the recorded source provenance; the [dependency manifest](third_party/manifest.json) provides the archive and file hashes for the exact contents shipped here. For lmms-eval, the revision was recorded in the source README.

Each archive contains its original `LICENSE` and a `NOTICE.release` describing the recorded adaptations. Environment setup extracts them under the corresponding `vendor/` directory, where those files remain available.

## Adaptations

- **SGLang:** DLM decoding, sampling, scheduling, attention and CUDA Graph integration, and checkpoint loading adaptations.
- **Transformers:** distribution changes to test-token defaults and local-kernel documentation examples.
- **lmms-eval:** a selected evaluation engine and OpenAI-compatible adapters, configurable dataset locations, request/response handling, and optional media dependencies. Task definitions are supplied by this repository's `eval/` directory.

The manifest records adapted file names. Existing copyright and license notices are preserved in those files.

## Derived Model Code

The VLM implementation includes code derived from Kimi K3 and other upstream model implementations. The Kimi K3 license is retained in [`models/vlm/LICENSE`](models/vlm/LICENSE); its [upstream license source](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/LICENSE) and the source-file notices provide the applicable terms and attribution.

## Scope

Additional dependencies installed by the environment recipes retain their own licenses and package notices. Model weights and datasets are distributed separately and retain their respective terms. Original project contributions are licensed under the [Apache License 2.0](LICENSE), with no additional restrictions imposed by this project. Third-party material and derived model code retain their applicable upstream licenses.

## Public Comparison Bases and Export Patches

The following publicly resolvable commits are comparison bases for reproducing the selected source exports. They are separate from the recorded snapshot revisions above; exact ancestry is not asserted. File-level patches describe added and replaced files, while unchanged files come from the public base. Replacement content is taken from the existing source archive to avoid distributing duplicate source trees.

- **transformers**: [v5.7.0](https://github.com/huggingface/transformers/commit/6ffbb07f93d9e44457450d1150136309b0dc966b) (`6ffbb07f93d9e44457450d1150136309b0dc966b`); [export patch](third_party/patches/transformers.json).
- **sglang**: [v0.5.6.post2](https://github.com/sgl-project/sglang/commit/5c8bd8b51b53b9b39eb1edec582ee43b21002106) (`5c8bd8b51b53b9b39eb1edec582ee43b21002106`); [export patch](third_party/patches/sglang.json).
- **lmms_eval**: [v0.5](https://github.com/EvolvingLMMs-Lab/lmms-eval/commit/8f142bc3082100dbb39aa9b15916c586f3237d09) (`8f142bc3082100dbb39aa9b15916c586f3237d09`); [export patch](third_party/patches/lmms_eval.json).

See [dependency sources](third_party/README.md) for rebuilding an export and checking all file hashes.

## Attribution and anonymous review

Third-party attribution for anonymous review

Names of upstream authors, companies and institutions, contact addresses,
repository namespaces, public model/dataset identifiers and example URLs in
third-party source and license notices identify their original sources. They
are retained for attribution and interoperability, not as a declaration of
this submission's authorship or affiliation. Such attribution alone does not
disclose submission authors. Original licenses and copyright notices remain
unchanged. Local adaptations are recorded separately; an unverified local
remark is not assigned to upstream merely because it is in a dependency.

## Model names and public examples

The released models use the GroundAnything and GroundAnything-VLM names.
References to Kimi K3 and Microsoft Mage in source notices identify the
upstream code and preserve attribution; they do not name these releases.
StreamMind is a retained optional component identifier.
