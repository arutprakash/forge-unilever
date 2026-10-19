STAGE: AUTHOR IMPLEMENTATION

The repository has already been cloned by a separate fetch container and mounted
read-only at /author-repo. Inspect its README and source. Use the authors'
actual model classes and training/evaluation code where possible, not a new
model implementation. Adapt its invocation to the neutral protocol in
/inputs/protocol.json, especially feature order, split boundaries, seed,
hyperparameters, complete test windows, and prediction export. This run measures
the authors' implementation under shared experimental controls; it is not
necessarily their original published training recipe.

Do not modify the read-only original. If compatibility or protocol changes are
needed, make a working copy under /workspace, apply minimal patches, and save a
unified patch against the original. Record every adaptation and the library
versions in implementation.md. Remove generated caches from the diff. Do not
look for or use generated implementations or other experiment results.

Keep the source's architecture unchanged. If it differs from the paper (for
example bias terms or initialization), record the difference without silently
changing it. Finish actual model training and prediction export for all models.
