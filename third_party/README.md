# Dependency Sources

This directory contains source snapshots of Transformers, SGLang, and lmms-eval used by the project. `manifest.json` records their versions, sources, installation paths, and file hashes.

Check and extract dependencies from the project root:

```bash
python scripts/prepare_dependencies.py
python scripts/prepare_dependencies.py --apply
```

Environment setup automatically extracts the required dependencies into `vendor/`. Existing directories are checked against the recorded hashes. Use a new environment directory when reinstalling.

Training and serving rely on extensions included in these snapshots. Use the provided environment configurations to install them. Each archive includes its original `LICENSE` and `NOTICE.release`. See [third-party notices](../THIRD_PARTY_NOTICES.md) for source revisions, license scope, and recorded adaptations.

## Reproduce a source export

Each `patches/<name>.json` pins a public comparison commit and its source archive checksum. These are verified comparison bases, not claims that the recorded snapshot revision is an upstream commit.

Download the `base.archive_url` from the selected recipe, then run:

```bash
python scripts/rebuild_dependency.py --name transformers --base-archive /path/to/upstream.tar.gz --output /path/to/rebuilt-transformers.tar.gz
```

The command verifies the base archive, patch recipe and bundled replacement content. It copies unchanged selected files from upstream, applies file-level additions/replacements from the bundled archive, and verifies every resulting file against `manifest.json`. Upstream files outside the manifest selection are omitted. Existing output files are never overwritten.

This patch format records hashes and operations without embedding removed private source or duplicating replacement source trees. It reproduces the distributed source files; it is not a patch for an entire upstream checkout. The original license and copyright notices remain in the exports.

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
